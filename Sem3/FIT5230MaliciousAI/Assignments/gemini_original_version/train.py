import os
import argparse
import json
import csv
import random
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset, ClassLabel
from sklearn.model_selection import train_test_split
from compressai.zoo import bmshj2018_hyperprior

# --- Configuration ---
# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Make your data directories relative to the script's location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
CONFIG_FILE = os.path.join(ARTIFACTS_DIR, "zed_config.json")
CALIBRATION_SCORES_FILE = os.path.join(ARTIFACTS_DIR, "calibration_scores.csv")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Core ZED Functions (Adapted from your notebook) ---

def choose_entropy_model(model_name: str = 'bmshj2018_hyperprior', quality: int = 8):
    """Loads a pretrained CompressAI model."""
    print(f"Loading model: {model_name} (quality={quality}) on {DEVICE}...")
    if model_name == 'bmshj2018_hyperprior':
        m = bmshj2018_hyperprior(quality=quality, pretrained=True, progress=True)
    else:
        # You can add other models like cheng2020_attn here
        raise ValueError(f'Unknown model_name: {model_name}')
    
    m.eval().to(DEVICE)
    try:
        m.update(force=True)
    except Exception as e:
        print(f'Warning: model.update() failed. This is expected for some models. Error: {e}')
    return m

def bpp_from_likelihoods(x: torch.Tensor, out: dict) -> torch.Tensor:
    """Estimates bits-per-pixel using model likelihoods."""
    N, _, H, W = x.shape
    num_pixels = N * H * W
    total_bits = torch.zeros_like(x, device=x.device).sum()
    
    for likelihood in out['likelihoods'].values():
        # Clamp likelihoods to avoid log(0) = -inf
        likelihood = torch.clamp(likelihood, min=1e-9)
        log_likelihood = -torch.log2(likelihood)
        total_bits += torch.sum(log_likelihood)
        
    return total_bits / num_pixels

def zed_score(image_path: str, model, multiscale=(1.0, 0.75, 0.5)) -> float:
    """Computes a ZED-style surprisal score for a single image."""
    scores = []
    try:
        base_image = Image.open(image_path).convert('RGB')
        for s in multiscale:
            w, h = base_image.size
            if s == 1.0:
                img = base_image
            else:
                new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
                img = base_image.resize((new_w, new_h), Image.LANCZOS)
            
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                out = model(x)
                bpp = bpp_from_likelihoods(x, out)
            scores.append(bpp.item())
    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")
        return -1.0 # Return an invalid score
    return np.mean(scores) if scores else -1.0

def batch_scores_with_checkpointing(paths: list, model, checkpoint_file: str) -> list:
    """Calculates scores for a list of images, with resume capability."""
    processed_scores = {}
    if os.path.exists(checkpoint_file):
        print(f"Loading existing scores from {checkpoint_file}...")
        with open(checkpoint_file, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in reader:
                processed_scores[row[0]] = float(row[1])
        print(f"Found {len(processed_scores)} already processed images.")

    paths_to_process = [p for p in paths if p not in processed_scores]
    
    if not paths_to_process:
        print("All images have already been scored.")
        return list(processed_scores.values())

    # Open file in append mode to add new results
    with open(checkpoint_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if the file is new/empty
        if not processed_scores:
            writer.writerow(['path', 'score_bpp'])

        for path in tqdm(paths_to_process, desc="Calculating ZED Scores"):
            score = zed_score(path, model)
            if score >= 0: # Only write valid scores
                processed_scores[path] = score
                writer.writerow([path, score])
                f.flush() # Ensure it's written immediately

    return list(processed_scores.values())

# --- Data Handling ---

def setup_directories():
    """Creates the necessary data and artifact directories."""
    os.makedirs(os.path.join(DATA_DIR, "train", "real"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "train", "fake"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val", "real"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "val", "fake"), exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    print("Created directory structure under data/ and artifacts/")

def download_and_save_data(dataset_name, num_samples, val_split, force_download=False):
    """Streams and saves a stratified subset of the dataset locally."""
    print(f"Preparing to download {num_samples} samples from '{dataset_name}'...")
    
    # Check if we can skip download
    if not force_download and os.path.exists(os.path.join(DATA_DIR, "download_complete.flag")):
        print("Dataset already downloaded and saved. Skipping. Use --force-download to override.")
        return

    # Load dataset stream
    dataset = load_dataset(dataset_name, split='train', streaming=True)
    dataset = dataset.shuffle(seed=42)

    print("Collecting and stratifying samples...")
    samples = []
    # Collect a larger pool to ensure stratification is meaningful
    pool_size = int(num_samples * 1.2) 
    for sample in tqdm(dataset.take(pool_size), total=pool_size, desc="Streaming samples"):
        samples.append(sample)

    # Separate by label
    real_samples = [s for s in samples if s['label'] == 'real']
    fake_samples = [s for s in samples if s['label'] == 'fake']

    # Take half from each class
    num_per_class = num_samples // 2
    selected_samples = random.sample(real_samples, num_per_class) + random.sample(fake_samples, num_per_class)
    random.shuffle(selected_samples)

    labels = [s['label'] for s in selected_samples]
    
    # Create train/validation split
    train_samples, val_samples, _, _ = train_test_split(
        selected_samples, labels, test_size=val_split, random_state=42, stratify=labels
    )

    def save_images(sample_set, split_name):
        print(f"Saving {len(sample_set)} images to data/{split_name}...")
        for i, item in enumerate(tqdm(sample_set, desc=f"Saving {split_name} set")):
            label = item['label']
            image = item['image']
            
            # Create a unique filename
            filename = f"{label}_{split_name}_{i:06d}.png"
            save_path = os.path.join(DATA_DIR, split_name, label, filename)
            
            if not os.path.exists(save_path):
                try:
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    image.save(save_path, "PNG")
                except Exception as e:
                    print(f"Could not save image {i}. Error: {e}")

    save_images(train_samples, "train")
    save_images(val_samples, "val")
    
    # Create a flag file to indicate completion
    with open(os.path.join(DATA_DIR, "download_complete.flag"), "w") as f:
        f.write("done")

# --- Main Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED Model Calibration Script")
    parser.add_argument("--dataset_name", type=str, default="Hemg/AI-Generated-vs-Real-Images-Datasets", help="Hugging Face dataset name.")
    parser.add_argument("--num_samples", type=int, default=100000, help="Total number of images to stream for train/val sets.")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of samples to use for validation set.")
    parser.add_argument("--model_name", type=str, default="bmshj2018_hyperprior", choices=['bmshj2018_hyperprior'], help="CompressAI model to use.")
    parser.add_argument("--quality", type=int, default=8, choices=range(1, 9), help="Model quality level (1-8).")
    parser.add_argument("--target_fpr", type=float, default=0.05, help="Target False Positive Rate for threshold calibration.")
    parser.add_argument("--force_download", action="store_true", help="Force re-downloading the dataset even if it exists.")
    args = parser.parse_args()

    # 1. Setup file structure
    setup_directories()
    
    # 2. Download and save data
    download_and_save_data(args.dataset_name, args.num_samples, args.val_split, args.force_download)

    # 3. Load the entropy model
    model = choose_entropy_model(args.model_name, args.quality)

    # 4. Calibrate threshold on REAL images only
    print("\n--- Starting Calibration ---")
    real_train_dir = os.path.join(DATA_DIR, "train", "real")
    real_paths = [os.path.join(real_train_dir, f) for f in os.listdir(real_train_dir)]
    
    if not real_paths:
        print(f"Error: No real images found in {real_train_dir}. Cannot calibrate.")
    else:
        # Calculate scores with checkpointing
        real_scores = batch_scores_with_checkpointing(real_paths, model, CALIBRATION_SCORES_FILE)
        
        # Calculate threshold
        threshold = float(np.quantile(real_scores, 1.0 - args.target_fpr))
        
        print(f"\nCalibration complete.")
        print(f"  - Real images used: {len(real_scores)}")
        print(f"  - Calibrated threshold @FPR~{args.target_fpr:.2f}: {threshold:.4f} bpp")

        # 5. Save the configuration for the evaluation script
        config = {
            "model_name": args.model_name,
            "quality": args.quality,
            "threshold_bpp": threshold,
            "calibrated_on_fpr": args.target_fpr,
            "num_calibration_samples": len(real_scores),
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved configuration to {CONFIG_FILE}")