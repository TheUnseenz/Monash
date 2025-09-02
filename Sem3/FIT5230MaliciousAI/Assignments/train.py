import os
import argparse
import json
import csv
import random
import torch
import numpy as np
# Import PIL.Image with an alias to avoid conflict
import PIL.Image as pil_image
from tqdm import tqdm
from datasets import load_dataset, ClassLabel, Image
from sklearn.model_selection import train_test_split
from compressai.zoo import bmshj2018_hyperprior
import sys
import io

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

def preprocess_image(image: pil_image.Image, target_size: int = 256) -> pil_image.Image:
    """
    Resizes and center-crops an image to a fixed square size.
    This eliminates dimensional differences between real and fake datasets.
    """
    # 1. Resize the image so the smaller side is `target_size`
    w, h = image.size
    if w < h:
        new_w = target_size
        new_h = int(h * (target_size / w))
    else:
        new_h = target_size
        new_w = int(w * (target_size / h))
    
    resized_image = image.resize((new_w, new_h), pil_image.LANCZOS)

    # 2. Center-crop to `target_size` x `target_size`
    left = (new_w - target_size) / 2
    top = (new_h - target_size) / 2
    right = (new_w + target_size) / 2
    bottom = (new_h + target_size) / 2

    cropped_image = resized_image.crop((left, top, right, bottom))
    return cropped_image

def zed_score(image_path: str, model) -> float:
    """Computes a ZED-style surprisal score on a preprocessed image."""
    try:
        base_image = pil_image.open(image_path).convert('RGB')
        
        # --- APPLY THE NEW UNIFORM PREPROCESSING ---
        processed_img = preprocess_image(base_image, target_size=256)
        
        arr = np.asarray(processed_img, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        
        with torch.inference_mode():
            out = model(x)
            bpp = bpp_from_likelihoods(x, out)
        return bpp.item()
        
    except Exception as e:
        print(f"Skipping {image_path} due to error: {e}")
        return -1.0 # Return an invalid score

# def zed_score(image_path: str, model, multiscale=(1.0, 0.75, 0.5)) -> float:
#     """Computes a ZED-style surprisal score for a single image."""
#     scores = []
#     required_factor = 64 # A common factor for deep learning models, adjust if needed
#     try:
#         base_image = pil_image.open(image_path).convert('RGB')
#         for s in multiscale:
#             w, h = base_image.size
#             if s == 1.0:
#                 img = base_image
#             else:
#                 new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
#                 img = base_image.resize((new_w, new_h), pil_image.LANCZOS)
            
#             # Pad the image to be a multiple of the required factor
#             w_padded = (img.width + required_factor - 1) // required_factor * required_factor
#             h_padded = (img.height + required_factor - 1) // required_factor * required_factor
            
#             # Create a new blank image and paste the original image onto it
#             padded_img = pil_image.new('RGB', (w_padded, h_padded))
#             padded_img.paste(img, (0, 0))
            
#             arr = np.asarray(padded_img, dtype=np.float32) / 255.0
#             x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            
#             with torch.inference_mode():
#                 out = model(x)
#                 bpp = bpp_from_likelihoods(x, out)
#             scores.append(bpp.item())
#     except Exception as e:
#         print(f"Skipping {image_path} due to error: {e}")
#         return -1.0 # Return an invalid score
#     return np.mean(scores) if scores else -1.0

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

    # Load dataset stream with manual image decoding
    dataset = load_dataset(dataset_name, split='train', streaming=True)

    # Disable automatic image decoding
    dataset = dataset.cast_column('image', Image(decode=False))

    dataset = dataset.shuffle(seed=42)

    print("Collecting and stratifying samples...")
    samples = []
    pool_size = int(num_samples * 1.2)
    # --- ADD A MINIMUM SIZE CONSTANT ---
    MIN_DIMENSION = 256


    for i, sample in enumerate(tqdm(dataset.take(pool_size), total=pool_size, desc="Streaming samples")):
        try:
            image_data = sample['image']['bytes']
            img = pil_image.open(io.BytesIO(image_data))
            
            # --- ADD THIS QUALITY FILTER ---
            w, h = img.size
            if w < MIN_DIMENSION or h < MIN_DIMENSION:
                continue # Skip this image, it's too small

            # # Check if the image data is in a bytes object
            # if 'bytes' in sample['image']:
            #     image_data = sample['image']['bytes']
            #     img = pil_image.open(io.BytesIO(image_data))
            # else:
            #     # If not bytes, try to open the image directly
            #     # This handles cases where 'image' is a raw file-like object
            #     img = pil_image.open(sample['image'])
            # Convert to RGBA first to handle transparency
            # Then, convert to RGB for your model which expects 3 channels
            img = img.convert('RGBA').convert('RGB')
            # Now, update the 'image' key in the sample dictionary
            sample['image'] = img

            # Check for the key and map the label
            if 'label' in sample:
                label = sample['label']
                if label == 0:
                    sample['label'] = 'real'
                elif label == 1:
                    sample['label'] = 'fake'
                
                # Check if the image is not None
                if sample['image'] is not None:
                    samples.append(sample)
                else:
                    print(f"\nWarning: Skipping sample at index {i} as it has no image data.")
            else:
                print(f"\nWarning: Skipping sample at index {i} as it is missing the 'label' key.")
        
        except (ZeroDivisionError, OSError) as e:
            # Catch ZeroDivisionError from malformed EXIF and OSError from bad images
            print(f"\nWarning: Skipping sample at index {i} due to a decoding error: {e}")
        except Exception as e:
            # Catch other potential errors, like malformed images
            print(f"\nWarning: Skipping sample at index {i} due to unexpected error: {e}")
    
        
        # Check for both 'is_fake' and 'label' keys
        # The Hemg dataset uses 'label', so we prioritize that
        if 'label' in sample:
            label = sample['label']
            # Map the integer label to your desired strings
            if label == 0:
                sample['label'] = 'real'
            elif label == 1:
                sample['label'] = 'fake'
            samples.append(sample)
        else:
            print(f"Warning: Skipping sample at index {i} as it is missing the 'label' key.")

    # Separate by label
    real_samples = [s for s in samples if s['label'] == 'real']
    fake_samples = [s for s in samples if s['label'] == 'fake']

    # Take half from each class, but ensure the number requested
    # is not larger than the population size.
    num_per_class = num_samples // 2
    
    # This is the critical change to prevent the ValueError
    num_real_to_take = min(num_per_class, len(real_samples))
    num_fake_to_take = min(num_per_class, len(fake_samples))
    
    if num_real_to_take == 0 or num_fake_to_take == 0:
        print("Not enough samples of one or both classes to proceed. Please adjust num_samples.")
        return
        
    selected_samples = random.sample(real_samples, num_real_to_take) + random.sample(fake_samples, num_fake_to_take)
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
    parser.add_argument("--low_percentile", type=float, default=0.05, help="Lower percentile for the two-sided threshold.")
    parser.add_argument("--high_percentile", type=float, default=0.95, help="Higher percentile for the two-sided threshold.")

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
        
        # Calculate BOTH thresholds
        threshold_low = float(np.quantile(real_scores, args.low_percentile))
        threshold_high = float(np.quantile(real_scores, args.high_percentile))
        
        print(f"\nCalibration complete.")
        print(f"  - Real images used: {len(real_scores)}")
        print(f"  - Low Threshold @{args.low_percentile:.2f} percentile: {threshold_low:.4f} bpp")
        print(f"  - High Threshold @{args.high_percentile:.2f} percentile: {threshold_high:.4f} bpp")

        # Save BOTH thresholds in the configuration
        config = {
            "model_name": args.model_name,
            "quality": args.quality,
            "threshold_low_bpp": threshold_low,
            "threshold_high_bpp": threshold_high,
            "num_calibration_samples": len(real_scores),
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved configuration to {CONFIG_FILE}")