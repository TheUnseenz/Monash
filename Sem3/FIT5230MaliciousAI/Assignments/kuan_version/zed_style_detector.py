# zed_stream_genimage.py
# Stream ~100k stratified samples from GenImage on HF, compute ZED-style surprisal (bpp),
# calibrate threshold on real images, and evaluate. No full dataset download required.

import os, math, random, io, json, itertools, collections
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Image as HFImage, IterableDataset
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

# -------- Config --------
HF_DATASET = "nebula/GenImage-arrow"   # community Arrow mirror with image + path columns; change if needed.  :contentReference[oaicite:4]{index=4}
SPLIT_NAME = "train"                   # dataset advertises one big 'train' split
RANDOM_SEED = 1337

# Target totals
N_TRAIN = 80_000
N_VAL   = 10_000
N_TEST  = 10_000
TARGET_TOTAL = N_TRAIN + N_VAL + N_TEST

# Class/strata targets: 50% real, 50% ai; within ai, balance generators
REAL_FRACTION = 0.5
AI_FRACTION   = 0.5

# CompressAI model config (pretrained entropy model for bpp/surprisal)
COMPRESSAI_MODEL = "cheng2020-attn"  # alternatives: "cheng2020-anchor", "mbt2018-mean", ...
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1    # compressai models are memory-heavy; keep small for safety
TARGET_FPR = 0.05 # threshold calibration target on REALs (5% FPR baseline)

# Optional image preprocessing
MAX_SIDE = 512  # resize max side to control compute, keep aspect; ZED uses multi-res encoder; we proxy with a fixed cap.


# -------- Utilities --------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_meta(image_path: str) -> Tuple[str, str, str]:
    """
    GenImage paths look like: 'ADM/train/ai/878_adm_130.PNG' or 'ImageNet/train/real/ILSVRC2012_...'
    Returns: (generator, split, label)
    """
    parts = image_path.split("/")
    if len(parts) < 3:
        return ("unknown", "unknown", "unknown")
    generator, split, label = parts[0], parts[1], parts[2]
    return generator, split, label  # label in {'ai','real'}

def is_real(label: str) -> bool:
    return label.lower() == "real"

def pil_from_ds(ex):
    # the dataset should expose an 'image' column; cast to HF Image and read PIL
    # Some mirrors store bytes/blob; datasets handles decoding if casted to Image feature.
    return ex["image"]  # already PIL if the feature is Image; else casted below

def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = max(w, h) / max_side if max(w, h) > max_side else 1.0
    if scale > 1.0:
        new_w, new_h = int(round(w/scale)), int(round(h/scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img

def to_tensor(img: Image.Image) -> torch.Tensor:
    # float tensor in [0,1], shape (1,C,H,W)
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:  # RGBA → RGB
        arr = arr[:, :, :3]
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return x

# -------- CompressAI scoring (bpp as surprisal) --------
@torch.no_grad()
def load_compressai_model(name: str):
    from compressai.zoo import load_model
    # quality doesn't matter for inference of likelihoods when using entropy models with default weights;
    # pick mid quality level—e.g., 3
    model = load_model(name, quality=3, pretrained=True).to(DEVICE).eval()
    return model

@torch.no_grad()
def bpp_score(imgs: List[Image.Image], model) -> List[float]:
    """
    Compute bits-per-pixel using CompressAI model negative log-likelihoods.
    This is a proxy to ZED's lossless surprisal; higher bpp ⇒ more 'surprising'.
    """
    scores = []
    for img in imgs:
        img = resize_max_side(img, MAX_SIDE)
        x = to_tensor(img).to(DEVICE)  # (1,C,H,W), [0,1]
        # CompressAI models expect 0..1 tensors, some models want multiples of 64; pad if needed
        _, _, H, W = x.shape
        pad_h = (64 - (H % 64)) % 64
        pad_w = (64 - (W % 64)) % 64
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        out = model(x)
        # 'likelihoods' contains y, z latents; total bits = sum(-log2 likelihoods)
        total_bits = 0.0
        num_pixels = x.shape[-1] * x.shape[-2]
        for lk in out["likelihoods"].values():
            total_bits += torch.sum(-torch.log2(lk)).item()
        # bpp is bits per input pixel (standard in learned compression)
        bpp = total_bits / num_pixels
        scores.append(float(bpp))
    return scores

# -------- Sampling (stratified, streaming) --------
@dataclass
class Quotas:
    train: int
    val: int
    test: int

def make_strata_quotas(total:int, ai_fraction:float, real_fraction:float,
                       ai_generators:List[str]) -> Dict[Tuple[str,str], Quotas]:
    """
    Strata keys are (label, generator_group).
      - real bucket: ('real','real')
      - ai buckets:  ('ai', gen) for each listed generator
    We apportion totals: real_fraction * total to real; ai_fraction * total across gens equally.
    """
    n_real = int(round(total * real_fraction))
    n_ai   = total - n_real
    per_ai = n_ai // max(1, len(ai_generators))
    leftovers = n_ai - per_ai * len(ai_generators)

    # 80/10/10 split per bucket
    def split3(n):
        n_train = int(round(n * 0.8))
        n_val   = int(round(n * 0.1))
        n_test  = n - n_train - n_val
        return Quotas(n_train, n_val, n_test)

    quotas = {("real", "real"): split3(n_real)}
    for i, g in enumerate(ai_generators):
        n = per_ai + (1 if i < leftovers else 0)
        quotas[("ai", g)] = split3(n)
    return quotas

def infer_ai_generators_from_path(path: str) -> str:
    gen, _, label = parse_meta(path)
    return gen

# Known GenImage generators from paper/homepage; presence may vary by mirror.
KNOWN_GENS = ["ADM","GLIDE","Wukong","VQDM","BigGAN","Midjourney","StableDiffusion","ImageNet"]  # 'ImageNet' appears for real; we remap

def map_stratum(path: str) -> Tuple[str,str]:
    gen, _, label = parse_meta(path)
    if is_real(label):
        return ("real","real")
    # unify generator names where needed
    g = gen
    if g.lower() in ["sd","stablediffusion","stable-diffusion"]:
        g = "StableDiffusion"
    if g.lower() == "imagenet":  # sometimes real paths begin with ImageNet
        return ("real","real")
    return ("ai", g)

def reservoir_take(iterable: Iterable, k: int, key=lambda _: True, rng: random.Random = random):
    """
    Classic reservoir sampling for k items matching a predicate key(item)=True.
    Returns a list of selected items.
    """
    res = []
    n = 0
    for item in iterable:
        if not key(item):
            continue
        n += 1
        if len(res) < k:
            res.append(item)
        else:
            j = rng.randint(1, n)
            if j <= k:
                res[j-1] = item
    return res


# -------- Main: stream → select → score → calibrate → eval --------
def main():
    set_seed(RANDOM_SEED)

    print(f"Loading HF dataset {HF_DATASET} (streaming)…")
    ds = load_dataset(HF_DATASET, split=SPLIT_NAME, streaming=True)
    # Ensure the 'image' column is decoded to PIL on the fly
    # Some mirrors already have it; to be safe:
    ds = ds.cast_column("image", HFImage())

    # Shuffle with a buffer so we don’t skim the head of shards
    ds = ds.shuffle(seed=RANDOM_SEED, buffer_size=50_000)

    # Build quotas: detect present AI generators on the fly (first lightweight pass over a small buffer)
    print("Probing generators…")
    probe = itertools.islice(ds, 50_000)  # peek a bit to see what gens exist
    present_gens = set()
    real_seen = False
    cached_probe = []
    for ex in probe:
        p = ex.get("image_path") or ex.get("path") or ""
        lab = parse_meta(p)[2]
        s = map_stratum(p)
        cached_probe.append(ex)
        if s[0] == "ai":
            present_gens.add(s[1])
        elif is_real(lab):
            real_seen = True
    if not real_seen:
        raise RuntimeError("Did not see any 'real' samples in the probe—check dataset columns.")
    ai_gens = sorted(list(present_gens)) or ["ADM","GLIDE","Wukong","VQDM","BigGAN","Midjourney","StableDiffusion"]
    print("Detected AI generators (from probe):", ai_gens)

    quotas = make_strata_quotas(TARGET_TOTAL, AI_FRACTION, REAL_FRACTION, ai_gens)
    counters = {k: Quotas(0,0,0) for k in quotas.keys()}
    # We’ll fill three lists per split with (md5, image_path) references
    selected = {"train": [], "val": [], "test": []}

    # Re-create a fresh shuffled stream that includes the probe items first
    def chain_probe_and_stream():
        for ex in cached_probe:
            yield ex
        # Resume streaming anew (shuffle again for variety)
        yield from load_dataset(HF_DATASET, split=SPLIT_NAME, streaming=True).cast_column("image", HFImage()).shuffle(seed=RANDOM_SEED+1, buffer_size=50_000)

    print("Selecting stratified ~100k samples by reservoir…")
    rng = random.Random(RANDOM_SEED)
    total_needed = sum(q.train+q.val+q.test for q in quotas.values())
    pbar = tqdm(total=total_needed, unit="img")
    for ex in chain_probe_and_stream():
        path = ex.get("image_path") or ex.get("path") or ""
        md5  = ex.get("md5") or path  # fallback to path as key
        label = parse_meta(path)[2].lower()
        s = map_stratum(path)
        if s not in quotas:
            # unseen generator—slot it into AI if label==ai (make dynamic bucket)
            if s[0] == "ai":
                quotas[s] = quotas[next(k for k in quotas if k[0]=="ai")]  # use any ai quota as template
                counters[s] = Quotas(0,0,0)
            else:
                continue

        q = quotas[s]
        c = counters[s]

        # fill train→val→test in that order
        assigned = False
        if c.train < q.train:
            selected["train"].append({"md5": md5, "image_path": path})
            counters[s] = Quotas(c.train+1, c.val, c.test); assigned = True
        elif c.val < q.val:
            selected["val"].append({"md5": md5, "image_path": path})
            counters[s] = Quotas(c.train, c.val+1, c.test); assigned = True
        elif c.test < q.test:
            selected["test"].append({"md5": md5, "image_path": path})
            counters[s] = Quotas(c.train, c.val, c.test+1); assigned = True

        if assigned:
            pbar.update(1)

        if (len(selected["train"]) + len(selected["val"]) + len(selected["test"])) >= total_needed:
            break

    pbar.close()
    for split in ["train","val","test"]:
        print(split, "selected:", len(selected[split]))

    # Build fast lookup sets
    lookup = {split: set((e["md5"], e["image_path"]) for e in selected[split]) for split in selected}

    # Helper to iterate a split by re-streaming and filtering by lookup
    def iter_split(split: str):
        wanted = lookup[split]
        # Use a moderate buffer shuffle to mitigate shard ordering
        stream = load_dataset(HF_DATASET, split=SPLIT_NAME, streaming=True).cast_column("image", HFImage()).shuffle(seed=RANDOM_SEED+42, buffer_size=20_000)
        for ex in stream:
            key = (ex.get("md5") or ex.get("image_path") or "", ex.get("image_path") or "")
            if key in wanted:
                yield ex

    # Load compressai model
    print(f"Loading CompressAI model: {COMPRESSAI_MODEL} on {DEVICE}")
    codec = load_compressai_model(COMPRESSAI_MODEL)

    def stream_scores(split: str):
        y_true, y_score = [], []
        batch_imgs = []
        batch_labels = []
        for ex in tqdm(iter_split(split), total=len(selected[split]), desc=f"Scoring {split}"):
            img = pil_from_ds(ex)
            path = ex.get("image_path") or ""
            lab = parse_meta(path)[2].lower()
            batch_imgs.append(img)
            batch_labels.append(0 if lab=="real" else 1)  # 0=real, 1=ai

            if len(batch_imgs) == BATCH_SIZE:
                scores = bpp_score(batch_imgs, codec)
                y_true.extend(batch_labels)
                y_score.extend(scores)
                batch_imgs, batch_labels = [], []
        if batch_imgs:
            scores = bpp_score(batch_imgs, codec)
            y_true.extend(batch_labels)
            y_score.extend(scores)
        return np.array(y_true), np.array(y_score)

    # Calibrate on TRAIN-REAL only for threshold @ target FPR
    print("Calibrating threshold on train REALs to achieve target FPR...")
    y_train, s_train = stream_scores("train")
    real_scores = s_train[y_train==0]
    thr = float(np.quantile(real_scores, 1.0 - TARGET_FPR))
    print(f"Calibrated threshold @FPR≈{TARGET_FPR:.2f}: {thr:.4f} bpp  (score>=thr ⇒ AI)")

    def evaluate(split: str, thr: float):
        y, s = stream_scores(split)
        y_pred = (s >= thr).astype(np.int32)
        acc = (y_pred == y).mean()
        try:
            auc = roc_auc_score(y, s)
        except ValueError:
            auc = float("nan")
        ap = average_precision_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        metrics = {
            "acc": float(acc),
            "auc": float(auc),
            "ap": float(ap),
            "thr": float(thr),
            "mean_bpp_real": float(np.mean(s[y==0])) if np.any(y==0) else None,
            "mean_bpp_ai": float(np.mean(s[y==1])) if np.any(y==1) else None,
            "n": int(len(y)),
            "pos_rate": float(np.mean(y)),
        }
        return metrics

    val_metrics = evaluate("val", thr)
    test_metrics = evaluate("test", thr)

    os.makedirs("zed_stream_out", exist_ok=True)
    with open("zed_stream_out/selection_summary.json","w") as f:
        json.dump({k: len(v) for k,v in selected.items()}, f, indent=2)
    with open("zed_stream_out/metrics.json","w") as f:
        json.dump({"val": val_metrics, "test": test_metrics, "thr": thr}, f, indent=2)

    print("\n=== Results ===")
    print("Threshold:", thr)
    print("Val:", val_metrics)
    print("Test:", test_metrics)
    print("Saved: zed_stream_out/metrics.json")

if __name__ == "__main__":
    main()
