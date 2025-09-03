"""
evaluate.py
- Load artifacts/zed_config.json & latest checkpoint
- Compute detection scores on ./data/test
- Calibrate thresholds on REAL images (default)
- Evaluate using a monotone score (distance-from-interval) so ROC/AUROC is interpretable
- Adversarial testing hook: modify input images in the "apply_adversarial_transform" function
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import PIL.Image as pil_image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix

from utils import (DATA_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, DEVICE,
                   preprocess_image_for_model, choose_entropy_model,
                   bpp_from_likelihoods_fallback, read_scores_csv, append_score_csv)

# --- Simple dataset reader for test folder ---
def list_test_images(test_dir: Path):
    real_dir = test_dir / "real"
    fake_dir = test_dir / "fake"
    real_paths = sorted([p for p in real_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    fake_paths = sorted([p for p in fake_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    return real_paths, fake_paths

# --- Adversarial hook: MODIFY HERE ---
def apply_adversarial_transform(img: pil_image.Image) -> pil_image.Image:
    """
    >>> MODIFY THIS FUNCTION to implement adversarial tests.
    Examples to try:
      - JPEG compression: save to BytesIO with low quality and re-open
      - Resize down and up (nearest) to simulate rescaling attack
      - Gaussian noise add
    Default: return image unchanged
    """
    return img

# --- monotone score helper ---
def distance_from_interval(scores: np.ndarray, low: float, high: float) -> np.ndarray:
    # 0 inside interval, positive outside; monotone with "more fake => larger"
    lo_diff = np.maximum(low - scores, 0.0)
    hi_diff = np.maximum(scores - high, 0.0)
    return np.maximum(lo_diff, hi_diff)

# --- main evaluation ---
def evaluate(args):
    config_path = Path(args.config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}; please run train.py first")

    with open(config_path, "r") as f:
        cfg = json.load(f)
    print("Loaded config:", cfg)

    # load model checkpoint
    ckpt_state, ckpt_path = None, None
    ckpt_files = list(Path("artifacts/checkpoints").glob("ckpt_*.pt"))
    if ckpt_files:
        ckpt_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        ckpt_path = ckpt_files[0]
        ckpt_state = torch.load(ckpt_path, map_location=DEVICE)
        print("Loaded checkpoint:", ckpt_path)
    else:
        print("No checkpoint found in artifacts/checkpoints. Evaluation may be impossible for the classifier model.")
        # continue: maybe user wants to only compute BPP-style scores

    # If classifier model checkpoint exists, reconstruct model architecture
    model = None
    if ckpt_state is not None:
        try:
            import timm
            backbone_name = ckpt_state["args"].get("backbone", "efficientnet_b4")
            backbone_pretrained = ckpt_state["args"].get("backbone_pretrained", True)
            backbone = timm.create_model(backbone_name, pretrained=backbone_pretrained, num_classes=0, global_pool="avg")
            feat_dim = backbone.num_features
            head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1)
            )
            model = nn.Sequential(backbone, head)
            model.load_state_dict(ckpt_state["model_state"])
            model.eval().to(DEVICE)
            print("Loaded classifier model for evaluation.")
        except Exception as e:
            print("Could not reconstruct classifier model from checkpoint:", e)
            model = None

    # Also attempt to load entropy model (SReC preferred)
    try:
        entropy_model, entropy_model_tag = choose_entropy_model(args.entropy_model, args.quality)
        print(f"Using entropy model: {entropy_model_tag}")
    except Exception as e:
        entropy_model = None
        entropy_model_tag = None
        print("No entropy model available:", e)

    # list test images
    test_dir = Path(args.test_data_dir)
    real_paths, fake_paths = list_test_images(test_dir)
    all_paths = list(real_paths) + list(fake_paths)
    y_true = np.array([0] * len(real_paths) + [1] * len(fake_paths))

    # scoring: if entropy_model available and it's SReC-like, compute D-statistic (placeholder).
    # If not, fallback to classifier logits (if model exists) or CompressAI bpp fallback.
    scores = []
    csv_scores_file = Path(ARTIFACTS_DIR) / "evaluation_scores.csv"
    existing = read_scores_csv(csv_scores_file)

    for p in tqdm(all_paths, desc="Scoring images"):
        if str(p) in existing:
            scores.append(existing[str(p)])
            continue
        try:
            img = pil_image.open(p).convert("RGB")
            img = apply_adversarial_transform(img)

            # If entropy model is SReC (lossless) and provides per-pixel NLL/H, we would compute D here.
            # We do a best-effort: if entropy_model_tag == "srec" and it has an API, use it.
            if entropy_model is not None and entropy_model_tag == "srec":
                # Placeholder example: depends on srec API -> user should replace with exact calls
                try:
                    x = np.asarray(img).astype(np.float32) / 255.0
                    x_t = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).to(DEVICE)
                    # [Unverified] API: the real SReC interface may differ greatly
                    out = entropy_model(x_t)
                    # assume out contains 'nll' and 'entropy' per-level arrays
                    nll0 = float(out["nll"][0].mean().item())
                    H0 = float(out["entropy"][0].mean().item())
                    D0 = nll0 - H0
                    score = abs(D0)
                except Exception as e:
                    # fallback to compressai scoring below
                    entropy_model = None
                    score = None
            else:
                score = None

            # fallback: if classifier model present, use sigmoid(logit) as "fake-likelihood score"
            if score is None and model is not None:
                t = transforms.Compose([
                    transforms.Lambda(lambda im: preprocess_image_for_model(im, target_size=cfg.get("target_size", 256))),
                    transforms.ToTensor()
                ])
                inp = t(img).unsqueeze(0).to(DEVICE)
                with torch.inference_mode():
                    logit = model(inp).cpu().numpy().squeeze().item()
                # produce a monotone "fakeness" score: higher = more likely fake
                score = float(torch.sigmoid(torch.tensor(logit)).item())

            # fallback: use CompressAI BPP-like score if entropy_model == 'bmshj' or entropy_model None
            if score is None and entropy_model is not None:
                try:
                    timg = preprocess_image_for_model(img, target_size=256)
                    arr = np.asarray(timg, dtype=np.float32)/255.0
                    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).to(DEVICE)
                    with torch.inference_mode():
                        out = entropy_model(x)
                        score_tensor = bpp_from_likelihoods_fallback(x, out)
                    score = float(score_tensor.item())
                except Exception:
                    score = -1.0

            if score is None:
                score = -1.0
            scores.append(score)
            append_score_csv(csv_scores_file, (str(p), float(score)))
        except Exception as e:
            scores.append(-1.0)
            append_score_csv(csv_scores_file, (str(p), -1.0))

    # filter invalid scores
    valid_indices = [i for i, s in enumerate(scores) if s != -1.0]
    if not valid_indices:
        print("No valid scores; aborting metrics.")
        return

    y_true_v = y_true[valid_indices]
    y_scores_v = np.array(scores)[valid_indices]

    # Calibration on REAL images only: compute low/high thresholds as percentiles of real scores
    real_mask = (y_true == 0)
    real_scores = np.array(scores)[real_mask.nonzero()] if len(real_mask.nonzero()[0])>0 else np.array([])
    if len(real_scores) == 0:
        print("No real images to calibrate thresholds.")
        low_thr, high_thr = None, None
    else:
        low_thr = float(np.quantile(real_scores, args.low_percentile))
        high_thr = float(np.quantile(real_scores, args.high_percentile))
        print(f"Calibrated thresholds on real: low={low_thr:.6f}, high={high_thr:.6f}")

    # Use monotone score for ROC: distance from [low, high]
    if low_thr is not None and high_thr is not None:
        score_for_roc = distance_from_interval(y_scores_v, low_thr, high_thr)
    else:
        # fallback: if classifier probabilities used, then larger means more fake already
        score_for_roc = y_scores_v.copy()

    # compute metrics
    pred = (score_for_roc > 0).astype(int)
    acc = accuracy_score(y_true_v, pred)
    f1 = f1_score(y_true_v, pred)
    try:
        auroc = roc_auc_score(y_true_v, score_for_roc)
    except Exception:
        auroc = float('nan')

    cm = confusion_matrix(y_true_v, pred)
    print("\n--- Evaluation ---")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, AUROC: {auroc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # Save final config with thresholds if applicable
    cfg["threshold_low_bpp"] = low_thr
    cfg["threshold_high_bpp"] = high_thr
    with open(Path(ARTIFACTS_DIR) / "zed_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print("Saved config with thresholds to artifacts/zed_config.json")

    # plot distributions
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.hist(y_scores_v[y_true_v==0], bins=60, alpha=0.7, label="Real")
    plt.hist(y_scores_v[y_true_v==1], bins=60, alpha=0.7, label="Fake")
    if low_thr is not None:
        plt.axvline(low_thr, linestyle="--", color="k", label="low_thr")
    if high_thr is not None:
        plt.axvline(high_thr, linestyle="--", color="k", label="high_thr")
    plt.legend()
    plt.title("Score distributions")

    plt.subplot(1,2,2)
    try:
        fpr, tpr, _ = roc_curve(y_true_v, score_for_roc)
        plt.plot(fpr, tpr, label=f"ROC (AUROC={auroc:.3f})")
        plt.plot([0,1],[0,1],"--",color="gray")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.legend()
    except Exception:
        plt.text(0.1, 0.5, "ROC not available")

    out_plot = Path(ARTIFACTS_DIR) / "evaluation_plots.png"
    plt.tight_layout()
    plt.savefig(out_plot)
    print("Saved plots to", out_plot)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ZED-like system")
    parser.add_argument("--config_path", type=str, default=os.path.join(ARTIFACTS_DIR, "zed_config.json"))
    parser.add_argument("--test_data_dir", type=str, default=os.path.join(DATA_DIR, "test"))
    parser.add_argument("--entropy_model", type=str, default="srec",
                        help="entropy model name to try to load (srec preferred).")
    parser.add_argument("--quality", type=int, default=8, help="quality for compressai fallback")
    parser.add_argument("--low_percentile", type=float, default=0.05)
    parser.add_argument("--high_percentile", type=float, default=0.95)
    args = parser.parse_args()

    evaluate(args)
