#!/usr/bin/env python3
"""
Train a tiny logistic head on top of ZED-style features (multi-scale bpp) while
streaming ~100k stratified samples from GenImage (no full download).

Features: bpp at scales [512, 384, 256] from a frozen CompressAI entropy model.
Classifier: single Linear layer (3->1), BCEWithLogitsLoss.

Checkpoints: saved to zed_stream_out/checkpoints/
 - best_val.pt           (best val AUC)
 - epoch_{E}.pt          (regular)
 - last.pt               (always latest)
Each includes model, optimizer, scaler, epoch, val_metrics.

Resume: --resume zed_stream_out/checkpoints/last.pt
"""

import os, math, random, json, itertools, argparse, time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, Image as HFImage
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# ---------------------- Config (defaults) ----------------------
HF_DATASET_DEFAULT = "nebula/GenImage-arrow"   # swap if needed
SPLIT_NAME = "train"
RANDOM_SEED = 1337
TOTAL = 1000
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
REAL_FRAC = 0.5
BATCH_SIZE_SCORE = 1            # scoring batch for compressai (keep small)
BATCH_SIZE_TRAIN = 256          # classifier batch size
LR = 1e-3
EPOCHS = 5
CHECKPOINT_EVERY_STEPS = 20   # classifier steps
TARGET_FPR = 0.05               # threshold calibrated on real-only (val)
SCALES = [512, 384, 256]        # multi-scale bpp features
COMPRESSAI_MODEL = "cheng2020-attn"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define the output directory relative to the script's directory
OUTDIR = os.path.join(script_dir, "zed_stream_out")
CKPT_DIR = os.path.join(OUTDIR, "checkpoints")
# OUTDIR = "zed_stream_out"
# CKPT_DIR = os.path.join(OUTDIR, "checkpoints")

# ---------------------- Utils ----------------------
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def parse_meta(path: str):
    parts = path.split("/")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]  # gen, split, label(ai/real)
    return "unknown","unknown","unknown"

def is_real_label(lbl: str) -> bool:
    return (lbl or "").lower() == "real"

def map_stratum(path: str) -> Tuple[str,str]:
    gen, _, label = parse_meta(path)
    if is_real_label(label): return ("real","real")
    g = gen
    if g.lower() in ["sd","stablediffusion","stable-diffusion"]:
        g = "StableDiffusion"
    if g.lower()=="imagenet": return ("real","real")
    return ("ai", g)

def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    if s <= max_side: return img
    scale = max_side / s
    nw, nh = int(round(w*scale)), int(round(h*scale))
    return img.resize((nw, nh), Image.LANCZOS)

def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim==2: arr = np.stack([arr,arr,arr],axis=-1)
    if arr.shape[2]==4: arr = arr[:,:,:3]
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)
    return x

@torch.no_grad()
def load_codec(name: str):
    from compressai.zoo import load_model
    m = load_model(name, quality=3, pretrained=True).to(DEVICE).eval()
    return m

@torch.no_grad()
def bpp_one(img: Image.Image, codec, max_side: int) -> float:
    im = resize_max_side(img, max_side)
    x = to_tensor(im).to(DEVICE)
    _, _, H, W = x.shape
    # pad to mult of 64
    ph, pw = (64 - H%64) % 64, (64 - W%64) % 64
    if ph or pw:
        x = torch.nn.functional.pad(x, (0,pw,0,ph), mode="replicate")
    out = codec(x)
    total_bits = 0.0
    num_pixels = x.shape[-1] * x.shape[-2]
    for lk in out["likelihoods"].values():
        total_bits += torch.sum(-torch.log2(lk)).item()
    return float(total_bits/num_pixels)

def feature_vector(img: Image.Image, codec) -> np.ndarray:
    return np.array([bpp_one(img, codec, s) for s in SCALES], dtype=np.float32)

# ---------------------- Stratified streaming selection ----------------------
@dataclass
class Quotas: train:int; val:int; test:int

def make_quotas(total:int, real_frac:float, ai_gens:List[str]) -> Dict[Tuple[str,str], Quotas]:
    n_real = int(round(total*real_frac))
    n_ai = total - n_real
    per_ai = n_ai // max(1,len(ai_gens))
    leftover = n_ai - per_ai*max(1,len(ai_gens))

    def split3(n):
        tr = int(round(n*TRAIN_FRAC))
        va = int(round(n*VAL_FRAC))
        te = n - tr - va
        return Quotas(tr,va,te)

    qs = {("real","real"): split3(n_real)}
    for i,g in enumerate(ai_gens):
        n = per_ai + (1 if i<leftover else 0)
        qs[("ai",g)] = split3(n)
    return qs

# ---------------------- Classifier ----------------------
class ZEDHead(nn.Module):
    def __init__(self, d=3):
        super().__init__()
        self.lin = nn.Linear(d, 1)
    def forward(self, x):
        return self.lin(x)  # logits

class Standardizer:
    """fit/transform for features; stored in checkpoints."""
    def __init__(self): self.mu=None; self.sig=None
    def fit(self, X: np.ndarray):
        self.mu = X.mean(0); self.sig = X.std(0)+1e-6
    def transform(self, X: np.ndarray):
        return (X - self.mu)/self.sig
    def state_dict(self): return {"mu": self.mu.tolist(), "sig": self.sig.tolist()}
    def load_state_dict(self, st): self.mu=np.array(st["mu"],dtype=np.float32); self.sig=np.array(st["sig"],dtype=np.float32)

# ---------------------- Training loop ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_dataset", default=HF_DATASET_DEFAULT)
    ap.add_argument("--total", type=int, default=TOTAL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--compressai_model", default=COMPRESSAI_MODEL)
    ap.add_argument("--checkpoint_every", type=int, default=CHECKPOINT_EVERY_STEPS)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    set_seed(args.seed)

    # stream dataset
    ds = load_dataset(args.hf_dataset, split=SPLIT_NAME, streaming=True)
    ds = ds.cast_column("image", HFImage()).shuffle(seed=args.seed, buffer_size=50_000)

    # probe to discover present generators
    print("Probing generators…")
    probe = list(itertools.islice(ds, 50_000))
    present_gens = set(); seen_real=False
    for ex in probe:
        p = ex.get("image_path") or ex.get("path") or ""
        lbl = parse_meta(p)[2]
        st = map_stratum(p)
        if st[0]=="ai": present_gens.add(st[1])
        if is_real_label(lbl): seen_real=True
    if not seen_real:
        raise RuntimeError("No real samples seen in probe; check dataset")
    ai_gens = sorted(list(present_gens)) or ["ADM","GLIDE","StableDiffusion","Midjourney","BigGAN","VQDM","Wukong"]

    quotas = make_quotas(args.total, REAL_FRAC, ai_gens)
    counts = {k: Quotas(0,0,0) for k in quotas}
    selected = {"train":[], "val":[], "test":[]}

    # chain probe back into stream
    def chain_stream():
        for ex in probe: yield ex
        more = load_dataset(args.hf_dataset, split=SPLIT_NAME, streaming=True)
        more = more.cast_column("image", HFImage()).shuffle(seed=args.seed+1, buffer_size=50_000)
        yield from more

    need = sum(q.train+q.val+q.test for q in quotas.values())
    print("Selecting ~{} examples…".format(need))
    picked = 0
    for ex in tqdm(chain_stream(), total=need*2):
        p = ex.get("image_path") or ex.get("path") or ""
        s = map_stratum(p)
        if s not in quotas:
            if s[0]=="ai":
                quotas[s] = quotas[next(k for k in quotas if k[0]=="ai")]
                counts[s] = Quotas(0,0,0)
            else:
                continue
        q = quotas[s]; c = counts[s]
        if c.train < q.train:
            selected["train"].append(ex); counts[s]=Quotas(c.train+1,c.val,c.test); picked+=1
        elif c.val < q.val:
            selected["val"].append(ex); counts[s]=Quotas(c.train,c.val+1,c.test); picked+=1
        elif c.test < q.test:
            selected["test"].append(ex); counts[s]=Quotas(c.train,c.val,c.test+1); picked+=1
        if picked>=need: break
    for sp in ["train","val","test"]:
        print(sp, len(selected[sp]))
    with open(os.path.join(OUTDIR,"selection_counts.json"), "w") as f:
        json.dump({k:len(v) for k,v in selected.items()}, f, indent=2)

    # codec
    print(f"Loading CompressAI model {args.compressai_model} on {DEVICE} …")
    codec = load_codec(args.compressai_model)

    # extract features/labels (note: streamed again but we already have ex with PIL in selected[])
    def to_xy(split):
        X, y = [], []
        for ex in tqdm(selected[split], desc=f"Featurizing {split}"):
            img = ex["image"]  # already PIL after cast
            path = ex.get("image_path") or ex.get("path") or ""
            lbl = 0 if is_real_label(parse_meta(path)[2]) else 1
            X.append(feature_vector(img, codec))
            y.append(lbl)
        return np.stack(X), np.array(y, dtype=np.int64)

    Xtr, ytr = to_xy("train")
    Xva, yva = to_xy("val")

    # standardize
    scaler = Standardizer()
    scaler.fit(Xtr)
    Xtr = scaler.transform(Xtr); Xva = scaler.transform(Xva)

    # torch loaders
    def make_loader(X, y, bs, shuffle):
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False)

    train_loader = make_loader(Xtr, ytr, BATCH_SIZE_TRAIN, True)
    val_loader   = make_loader(Xva, yva, BATCH_SIZE_TRAIN, False)

    # model/opt
    model = ZEDHead(d=len(SCALES)).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    # resume?
    start_epoch = 0; global_step = 0; best_auc = -1.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0)
        best_auc = ckpt.get("best_auc", -1.0)
        global_step = ckpt.get("global_step", 0)
        print(f"Resumed from {args.resume} (epoch {start_epoch}, best_auc {best_auc:.4f})")

    # helpers
    def save_ckpt(name, epoch, val_stats, best_auc, global_step):
        path = os.path.join(CKPT_DIR, name)
        torch.save({
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "val_stats": val_stats,
            "best_auc": best_auc,
            "global_step": global_step,
            "config": {
                "scales": SCALES,
                "compressai_model": args.compressai_model,
                "hf_dataset": args.hf_dataset
            }
        }, path)
        print(f"[ckpt] saved {path}")

    def eval_loader(loader):
        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                logits = model(xb).squeeze(1)
                prob = torch.sigmoid(logits)
                ys.append(yb.cpu().numpy()); ps.append(prob.cpu().numpy())
        y = np.concatenate(ys); p = np.concatenate(ps)
        auc = roc_auc_score(y, p) if len(np.unique(y))>1 else float("nan")
        ap = average_precision_score(y, p)
        # Optional threshold by target FPR on REALS (y==0):
        thr = None
        reals = p[y==0]
        if len(reals)>10:
            thr = float(np.quantile(reals, 1.0 - TARGET_FPR))
        acc = np.mean((p >= (thr if thr is not None else 0.5)) == y)
        return {"auc": float(auc), "ap": float(ap), "acc": float(acc), "thr": thr}

    # train
    print("Training…")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            global_step += 1
            if global_step % args.checkpoint_every == 0:
                val_stats = eval_loader(val_loader)
                save_ckpt(f"step_{global_step}.pt", epoch, val_stats, best_auc, global_step)

        # epoch end
        val_stats = eval_loader(val_loader)
        print(f"[epoch {epoch+1}/{args.epochs}] val AUC={val_stats['auc']:.4f} AP={val_stats['ap']:.4f} ACC={val_stats['acc']:.4f} thr={val_stats['thr']}")
        save_ckpt("last.pt", epoch+1, val_stats, best_auc, global_step)
        if np.isnan(val_stats["auc"])==False and val_stats["auc"] > best_auc:
            best_auc = val_stats["auc"]
            save_ckpt("best_val.pt", epoch+1, val_stats, best_auc, global_step)

    # also score test split quickly for a snapshot
    print("Computing quick test snapshot…")
    # reuse codec features (already selected/test in memory)
    def to_xy(split):
        X, y = [], []
        for ex in tqdm(selected[split], desc=f"Featurizing {split}"):
            img = ex["image"]; path = ex.get("image_path") or ex.get("path") or ""
            lbl = 0 if is_real_label(parse_meta(path)[2]) else 1
            X.append(feature_vector(img, codec)); y.append(lbl)
        return np.stack(X), np.array(y, dtype=np.int64)
    Xte, yte = to_xy("test")
    Xte = scaler.transform(Xte)
    te_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte).float()),
        batch_size=BATCH_SIZE_TRAIN, shuffle=False)
    test_stats = eval_loader(te_loader)
    with open(os.path.join(OUTDIR, "metrics_train_val_test.json"), "w") as f:
        json.dump({"val": ckpt.get("val_stats") if (args.resume and 'ckpt' in locals()) else None,
                   "final_val": val_stats,
                   "test": test_stats}, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
