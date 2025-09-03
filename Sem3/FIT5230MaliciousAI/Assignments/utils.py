"""
utils.py
Helpers used by train.py and evaluate.py

Responsibilities:
- dataset streaming & saving into ./data/
- model selection (try SReC lossless; fallback to CompressAI)
- scoring functions + preprocessing
- checkpoint helpers
"""

import os
import io
import json
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
from tqdm import tqdm
import PIL.Image as pil_image

# Torch-related imports are placed here so callers can import utils without causing hard failures
import torch

# --- Directory utilities ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Default image preprocessing size (used only for the fallback/compatibility)
DEFAULT_TARGET_SIZE = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset streaming & saving ---
def download_streamed_dataset(hf_dataset_name: str,
                              out_base: Path,
                              num_samples: int = 100_000,
                              val_split: float = 0.1,
                              test_split: float = 0.1,
                              min_dimension: int = 256,
                              seed: int = 42,
                              force_download: bool = False) -> None:
    """
    Stream a dataset from Hugging Face datasets and save a stratified subset to disk:
      out_base/
        train/real
        train/fake
        val/real
        val/fake
        test/real
        test/fake

    - hf_dataset_name: string like "GenImage/whatever" or "WildFake/whatever"
    - This function will try to detect label keys 'label' or 'is_fake' and map to 'real'/'fake'.
    - Will show progress and create a flag file to resume.
    """
    from datasets import load_dataset, Image

    out_base = Path(out_base)
    assert out_base.exists() or out_base.mkdir(parents=True, exist_ok=True)

    flag_file = out_base / ".download_complete.flag"
    if flag_file.exists() and not force_download:
        print(f"Dataset already downloaded to {out_base}. Use force_download=True to override.")
        return

    
    print(f"Streaming dataset '{hf_dataset_name}' until {num_samples:,} usable images are saved to {out_base}")
    stream = load_dataset(hf_dataset_name, split="train", streaming=True)
    try:
        stream = stream.cast_column("image", Image(decode=False))
    except Exception:
        pass

    attempts = 0
    samples = []
    MIN_DIM = min_dimension
    rng = random.Random(seed)

    # tqdm with no fixed total, it will just keep going
    for item in tqdm(stream, desc="Streaming HF dataset", unit="sample"):
        if len(samples) >= num_samples or attempts > num_samples*10:
            break
    # print(f"Streaming dataset '{hf_dataset_name}' and saving {num_samples:,} images to {out_base}")
    # stream = load_dataset(hf_dataset_name, split="train", streaming=True)
    # # try to ensure we don't auto-decode images (we will handle bytes)
    # try:
    #     stream = stream.cast_column("image", Image(decode=False))
    # except Exception:
    #     # some datasets may not have 'image' or casting fails; keep going
    #     pass

    # sample_pool = int(num_samples * 1.2)
    # samples = []
    # MIN_DIM = min_dimension
    # rng = random.Random(seed)

    # for i, item in enumerate(tqdm(stream.take(sample_pool), total=sample_pool, desc="Streaming HF dataset")):
        # if len(samples) >= sample_pool:
        #     break
        try:
            # access image bytes
            if isinstance(item.get("image", None), dict) and "bytes" in item["image"]:
                img_bytes = item["image"]["bytes"]
                pil = pil_image.open(io.BytesIO(img_bytes)).convert("RGBA").convert("RGB")
            else:
                # fallback: try direct PIL open if possible
                img_candidate = item.get("image", None)
                if img_candidate is None:
                    continue
                try:
                    pil = pil_image.open(io.BytesIO(img_candidate)).convert("RGBA").convert("RGB")
                except Exception:
                    # maybe it's already a PIL-like object
                    pil = pil_image.Image.open(item["image"]).convert("RGBA").convert("RGB")

            w, h = pil.size
            if w < MIN_DIM or h < MIN_DIM:
                continue

            # find label -- dataset dependent
            label = None
            if "label" in item:
                label = item["label"]
            elif "is_fake" in item:
                label = item["is_fake"]
            elif "target" in item:
                label = item["target"]
            # else we try to skip items w/o label
            if label is None:
                continue

            # Normalize label into 'real'/'fake'
            if isinstance(label, (int, np.integer)):
                # assume 0 = real, 1 = fake in many datasets; if not, user should double-check
                label_str = "real" if int(label) == 0 else "fake"
            elif isinstance(label, str):
                # many datasets have 'real'/'fake' already
                label_str = label.lower()
                if label_str not in ("real", "fake"):
                    # try some heuristics
                    if "fake" in label_str or "synth" in label_str:
                        label_str = "fake"
                    else:
                        label_str = "real"
            else:
                # fallback
                label_str = "real"

            samples.append({"image": pil, "label": label_str})
        except Exception:
            # skip malformed images
            continue

    # split by label
    real_samples = [s for s in samples if s["label"] == "real"]
    fake_samples = [s for s in samples if s["label"] == "fake"]
    num_per_class = num_samples // 2

    if len(real_samples) < 10 or len(fake_samples) < 10:
        print("Warning: Not enough samples collected for one of the classes; collected counts:",
              len(real_samples), len(fake_samples))

    num_real = min(len(real_samples), num_per_class)
    num_fake = min(len(fake_samples), num_per_class)
    if num_real == 0 or num_fake == 0:
        raise RuntimeError("Could not collect sufficient samples for both classes. Try a different dataset.")

    selected = rng.sample(real_samples, num_real) + rng.sample(fake_samples, num_fake)
    rng.shuffle(selected)

    # Train / Val / Test split
    labels = [s["label"] for s in selected]
    from sklearn.model_selection import train_test_split
    train_val, test_samples = train_test_split(selected, test_size=test_split, stratify=labels, random_state=seed)
    train_samples, val_samples = train_test_split(train_val, test_size=val_split / (1 - test_split),
                                                  stratify=[s["label"] for s in train_val], random_state=seed)

    def save_set(items, split_name):
        out_dir = out_base / split_name
        for s in items:
            p = out_dir / s["label"]
            p.mkdir(parents=True, exist_ok=True)
        print(f"Saving {len(items)} items to {out_base}/{split_name}/ (this may take a while)")
        for i, s in enumerate(tqdm(items, desc=f"Saving {split_name}")):
            fname = f"{s['label']}_{split_name}_{i:06d}.png"
            out_path = out_base / split_name / s["label"] / fname
            if not out_path.exists():
                try:
                    s["image"].save(out_path, "PNG")
                except Exception as e:
                    # continue on failure
                    continue

    save_set(train_samples, "train")
    save_set(val_samples, "val")
    save_set(test_samples, "test")

    # write a small metadata file
    meta = {
        "dataset_name": hf_dataset_name,
        "num_requested": num_samples,
        "num_saved": len(selected),
        "splits": {"train": len(train_samples), "val": len(val_samples), "test": len(test_samples)}
    }
    with open(out_base / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # create flag
    with open(flag_file, "w") as f:
        f.write("done")

    print("Dataset save complete.")


# --- Preprocessing and padding helpers ---
def preprocess_image_for_model(img: pil_image.Image, target_size: int = DEFAULT_TARGET_SIZE) -> pil_image.Image:
    """
    Resize and center-crop to target_size x target_size.
    NOTE: If you later use a lossless model that expects originals, consider bypassing this.
    """
    w, h = img.size
    # resize smaller side to target_size, preserve aspect, then center crop
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))

    resized = img.resize((new_w, new_h), pil_image.LANCZOS)
    left = (new_w - target_size) / 2
    top = (new_h - target_size) / 2
    right = (new_w + target_size) / 2
    bottom = (new_h + target_size) / 2
    cropped = resized.crop((left, top, right, bottom))
    return cropped


def pad_to_multiple(img: pil_image.Image, factor: int = 64) -> pil_image.Image:
    """Pad bottom/right with black pixels so width & height are multiples of factor."""
    w, h = img.size
    pad_w = (factor - w % factor) % factor
    pad_h = (factor - h % factor) % factor
    if pad_w == 0 and pad_h == 0:
        return img
    new = pil_image.new("RGB", (w + pad_w, h + pad_h), (0, 0, 0))
    new.paste(img, (0, 0))
    return new

# --- Model selection ---
def choose_entropy_model(model_name: str = "srec", quality: int = 8):
    """
    Try to load a lossless model (SReC) if available. Otherwise fallback to CompressAI's bmshj2018_hyperprior.
    Return: model object and a string 'srec' or 'bmshj' so caller knows which path is used.

    NOTE: [Unverified] ZED uses a lossless codec (SReC). If you don't have SReC installed,
    the fallback is NOT equivalent to ZED. This function clearly warns when fallback occurs.
    """
    # Preferred: SReC / any lossless learned model that provides per-pixel log-probabilities
    try:
        import srec  # [Unverified] assumes an srec package is installed
        # The exact loading depends on srec's API. This is a placeholder example.
        model = srec.load_pretrained().to(DEVICE)
        model.eval()
        return model, "srec"
    except Exception:
        print("[Warning] SReC (lossless model) not available. Falling back to CompressAI (lossy) model.")
        try:
            from compressai.zoo import bmshj2018_hyperprior
        except Exception as e:
            raise ImportError("Neither SReC nor CompressAI models are importable. Install compressai or srec.") from e
        model = bmshj2018_hyperprior(quality=quality, pretrained=True, progress=True).eval().to(DEVICE)
        # compressai models often require update()
        try:
            model.update(force=True)
        except Exception:
            pass
        return model, "bmshj"


def bpp_from_likelihoods_fallback(x: torch.Tensor, out: Dict) -> torch.Tensor:
    """
    Compute bits-per-pixel from CompressAI's 'likelihoods' structure.
    This is the fallback method; it is NOT equivalent to ZED's lossless D() statistic.
    """
    N, _, H, W = x.shape
    num_pixels = N * H * W
    total_bits = torch.tensor(0.0, device=x.device)
    for likelihood in out.get("likelihoods", {}).values():
        likelihood = torch.clamp(likelihood, min=1e-9)
        total_bits = total_bits + torch.sum(-torch.log2(likelihood))
    return total_bits / num_pixels


# --- Checkpoint helpers ---
def save_training_checkpoint(state: dict, fname: str):
    path = CHECKPOINT_DIR / fname
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_latest_checkpoint() -> Optional[Tuple[dict, Path]]:
    # returns (state, path) or (None, None)
    ckpts = list(CHECKPOINT_DIR.glob("ckpt_*.pt"))
    if not ckpts:
        return None, None
    ckpts_sorted = sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
    latest = ckpts_sorted[0]
    state = torch.load(latest, map_location=DEVICE)
    return state, latest

# --- CSV checkpointing for scoring ---
def read_scores_csv(path: Path) -> Dict[str, float]:
    out = {}
    if not path.exists():
        return out
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            out[row[0]] = float(row[1])
    return out

def append_score_csv(path: Path, row: Tuple[str, float]):
    header_needed = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["path", "score_bpp"])
        w.writerow([row[0], row[1]])
