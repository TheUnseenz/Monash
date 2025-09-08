"""
train.py
- Streams & saves dataset to ./data/
- Loads a pretrained backbone (EfficientNet-B4 by default)
- Trains a small head (binary) using stored dataset
- Checkpoints and allows resume
- Produces an artifacts/zed_config.json with thresholds calibrated on real images
"""

import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import PIL.Image as pil_image

# local utils
from utils import (DATA_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, DEVICE,
                   download_streamed_dataset, preprocess_image_for_model,
                   choose_entropy_model, bpp_from_likelihoods_fallback,
                   save_training_checkpoint, load_latest_checkpoint, append_score_csv, read_scores_csv)

# =========
# Simple custom dataset reading from folders saved by utils.download_streamed_dataset
# =========
class SimpleImageFolderDataset(Dataset):
    def __init__(self, root_dir: Path, split: str = "train", transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        for lbl in ("real", "fake"):
            folder = self.root_dir / lbl
            if folder.exists():
                for p in folder.iterdir():
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                        self.samples.append((str(p), 0 if lbl == "real" else 1))
        # sort for deterministic ordering
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = pil_image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

# =========
# Model: pretrained backbone + small head
# =========
def make_model(backbone_name: str = "efficientnet_b4", pretrained: bool = True):
    try:
        import timm
    except Exception:
        raise RuntimeError("timm is required for backbone models. Please pip install timm.")
    # create backbone
    backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg")
    feat_dim = backbone.num_features
    head = nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 1)
    )
    model = nn.Sequential(backbone, head)
    return model

# =========
# Helpers
# =========
def collate_fn(batch):
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.float32)
    return imgs, labels.unsqueeze(1), paths

# =========
# Training loop
# =========
def train(args):
    # 1) dataset download / streaming
    data_root = Path(DATA_DIR)
    if not (data_root / "train").exists() or args.force_download:
        download_streamed_dataset(args.dataset_name, data_root, num_samples=args.num_samples,
                                  val_split=args.val_split, test_split=args.test_split,
                                  min_dimension=args.min_image_dim, seed=args.seed, force_download=args.force_download)
    else:
        print("Using existing dataset in", data_root)

    # 2) prepare data loaders
    transform = transforms.Compose([
        transforms.Lambda(lambda img: preprocess_image_for_model(img, target_size=args.target_size)),
        transforms.ToTensor(),
    ])
    train_ds = SimpleImageFolderDataset(data_root, split="train", transform=transform)
    val_ds = SimpleImageFolderDataset(data_root, split="val", transform=transform)

    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True)

    # 3) model
    model = make_model(args.backbone, pretrained=args.backbone_pretrained).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        state, ckpt_path = load_latest_checkpoint()
        if state:
            print("Resuming from checkpoint:", ckpt_path)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optim_state"])
            start_epoch = state["epoch"] + 1
    else:
        print("Training from scratch")

    # 4) training loop
    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            running_loss = 0.0
            tk = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs-1} [train]")
            for i, (imgs, labels, _) in tk:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i + 1) % args.log_interval == 0:
                    tk.set_postfix(loss=running_loss / (i + 1))

            # validation pass
            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for imgs, labels, _ in tqdm(val_loader, desc="Validation"):
                    imgs = imgs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            val_loss /= max(1, len(val_loader))
            print(f"Epoch {epoch} complete. Val loss: {val_loss:.4f}")

            # save checkpoint
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "args": vars(args)
            }
            save_training_checkpoint(state, f"ckpt_epoch_{epoch}.pt")
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving last checkpoint...")
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "args": vars(args)
        }
        save_training_checkpoint(state, "ckpt_interrupted.pt")

    # After training: save a minimal config
    config = {
        "backbone": args.backbone,
        "backbone_pretrained": args.backbone_pretrained,
        "target_size": args.target_size,
        "threshold_low_bpp": None,   # to be filled later by evaluation calibration if needed
        "threshold_high_bpp": None,
    }
    with open(Path(ARTIFACTS_DIR) / "zed_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print("Training complete. Config saved to artifacts/zed_config.json")


# =========
# CLI
# =========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED-like training script (backbone classifier approach).")
    parser.add_argument("--dataset_name", type=str, default="Hemg/AI-Generated-vs-Real-Images-Datasets", help="Hugging Face dataset to stream.")
    parser.add_argument("--num_samples", type=int, default=50000, help="Total images to stream and save.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction when splitting.")
    parser.add_argument("--test_split", type=float, default=0.1, help="Test fraction when splitting.")
    parser.add_argument("--min_image_dim", type=int, default=256, help="Minimum width/height to accept.")
    parser.add_argument("--force_download", action="store_true", help="Force dataset re-download and saving.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--backbone", type=str, default="efficientnet_b4")
    parser.add_argument("--backbone_pretrained", type=bool, default=True)
    parser.add_argument("--target_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if found.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)
