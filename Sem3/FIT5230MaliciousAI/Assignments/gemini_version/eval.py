import os
import argparse
import json
import csv
import torch
import numpy as np
import warnings
import PIL.Image as pil_image
import PIL.ImageOps as pil_imageops
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix

# Suppress harmless warnings from matplotlib
warnings.filterwarnings("ignore", module="matplotlib")

# --- Configuration ---
# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Make your data directories relative to the script's location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
DEFAULT_TEST_DIR = os.path.join(DATA_DIR, "val") # Use validation set for evaluation by default
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Core ZED Functions (Copied from train.py for standalone use) ---

# Note: In a larger project, these would be in a shared utils.py file.
def choose_entropy_model(model_name: str, quality: int):
    # This dynamic import is to avoid a hard dependency on compressai if just viewing results
    try:
        from compressai.zoo import bmshj2018_hyperprior
    except ImportError:
        print("Error: compressai is not installed. Please run 'pip install compressai'")
        exit()

    print(f"Loading model: {model_name} (quality={quality}) on {DEVICE}...")
    if model_name == 'bmshj2018_hyperprior':
        m = bmshj2018_hyperprior(quality=quality, pretrained=True, progress=True)
    else:
        raise ValueError(f'Unknown model_name: {model_name}')

    m.eval().to(DEVICE)
    try:
        m.update(force=True)
    except Exception as e:
        pass
    return m

def bpp_from_likelihoods(x: torch.Tensor, out: dict) -> torch.Tensor:
    N, _, H, W = x.shape
    num_pixels = N * H * W
    total_bits = torch.zeros_like(x, device=x.device).sum()
    for likelihood in out['likelihoods'].values():
        likelihood = torch.clamp(likelihood, min=1e-9)
        total_bits += torch.sum(-torch.log2(likelihood))
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

def zed_score_from_image(image: pil_image.Image, model) -> float:
    """
    Computes a ZED-style surprisal score from a PIL Image object
    after uniform preprocessing.
    """
    # --- APPLY THE NEW UNIFORM PREPROCESSING ---
    # This ensures all images are 256x256 before scoring
    processed_img = preprocess_image(image, target_size=256)
    
    arr = np.asarray(processed_img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        out = model(x)
        bpp = bpp_from_likelihoods(x, out)
        
    return bpp.item()

# def zed_score_from_image(image: pil_image.Image, model, multiscale=(1.0, 0.75, 0.5)) -> float:
#     """Computes a ZED-style surprisal score from a PIL Image object."""
#     scores = []
#     required_factor = 64
#     try:
#         base_image = image.convert('RGB')
#         for s in multiscale:
#             w, h = base_image.size
#             if s == 1.0:
#                 img = base_image
#             else:
#                 new_w, new_h = max(1, int(w * s)), max(1, int(h * s))
#                 img = base_image.resize((new_w, new_h), pil_image.LANCZOS)

#             w_padded = (img.width + required_factor - 1) // required_factor * required_factor
#             h_padded = (img.height + required_factor - 1) // required_factor * required_factor

#             padded_img = pil_image.new('RGB', (w_padded, h_padded))
#             padded_img.paste(img, (0, 0))

#             arr = np.asarray(padded_img, dtype=np.float32) / 255.0
#             x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

#             with torch.inference_mode():
#                 out = model(x)
#                 bpp = bpp_from_likelihoods(x, out)
#             scores.append(bpp.item())
#         return np.mean(scores) if scores else -1.0
#     except Exception as e:
#         print(f"Error in zed_score_from_image: {e}")
#         return -1.0
        
# --- Adversarial Test Section ---
def apply_adversarial_transform(image: pil_image.Image) -> pil_image.Image:
    """
    !!! ENTRY POINT FOR ADVERSARIAL MODIFICATIONS !!!
    Modify the input PIL image here to test robustness.
    By default, it does nothing. Uncomment examples to apply them.
    """
    # --- Example 1: JPEG Compression ---
    # from io import BytesIO
    # buffer = BytesIO()
    # image.save(buffer, format="JPEG", quality=75)
    # buffer.seek(0)
    # return pil_image.open(buffer)

    # --- Example 2: Resizing ---
    # original_size = image.size
    # downscaled_size = (original_size[0] // 2, original_size[1] // 2)
    # image = image.resize(downscaled_size, pil_image.NEAREST)
    # image = image.resize(original_size, pil_image.LANCZOS) # Upscale back
    # return image

    # --- Example 3: Adding Gaussian Noise ---
    # img_np = np.array(image).astype(np.float32)
    # noise = np.random.normal(0, 15, img_np.shape) # std dev = 15
    # noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    # return pil_image.fromarray(noisy_img_np)

    # Default: No transformation
    return image


# --- Evaluation Logic ---
def evaluate_model(model, config, test_data_dir, results_csv):
    """Runs evaluation on the test set and saves scores and metrics."""
    # Check for the dual-thresholds
    try:
        low_threshold = config['threshold_low_bpp']
        high_threshold = config['threshold_high_bpp']
    except KeyError:
        print("Error: Config file is missing 'threshold_low_bpp' or 'threshold_high_bpp'.")
        print("Please re-run train.py with the updated script.")
        return

    real_dir = os.path.join(test_data_dir, "real")
    fake_dir = os.path.join(test_data_dir, "fake")

    if not os.path.isdir(real_dir) or not os.path.isdir(fake_dir):
        print(f"Error: Test directory '{test_data_dir}' must contain 'real' and 'fake' subdirectories.")
        return

    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
    
    all_paths = real_paths + fake_paths
    true_labels = [0] * len(real_paths) + [1] * len(fake_paths) # 0=real, 1=fake

    scores = []
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label', 'score_bpp', 'predicted_label'])
        
        for path, label in tqdm(zip(all_paths, true_labels), total=len(all_paths), desc="Evaluating Test Set"):
            try:
                # Correctly handle the potential name conflict from the 'Image' import
                img = pil_image.open(path)
                
                # Apply any defined transformations for robustness testing
                transformed_img = apply_adversarial_transform(img)

                score = zed_score_from_image(transformed_img, model)
                # Predict FAKE if score is outside the calibrated normal range
                prediction = 1 if score < low_threshold or score > high_threshold else 0
                
                scores.append(score)
                writer.writerow([path, label, score, prediction])
            except Exception as e:
                print(f"Could not process {path}. Error: {e}")
                # Add placeholders for failed images
                scores.append(-1) 
                writer.writerow([path, label, -1, -1])

    # Filter out failed images for metric calculation
    valid_indices = [i for i, s in enumerate(scores) if s != -1]
    if not valid_indices:
        print("No images could be processed. Cannot calculate metrics.")
        return
        
    y_true = np.array(true_labels)[valid_indices]
    y_scores = np.array(scores)[valid_indices]
    # The previous `y_pred` line was incorrect as it was based on only a single threshold
    y_pred = (y_scores < low_threshold) | (y_scores > high_threshold)
    # Convert boolean predictions to integers (True=1, False=0)
    y_pred = y_pred.astype(int)

    # --- Calculate and Display Metrics ---
    print("\n--- Evaluation Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUROC: {auroc:.4f}")
    print("Confusion Matrix:")
    print(f"  [[TN: {cm[0][0]}, FP: {cm[0][1]}]")
    print(f"   [FN: {cm[1][0]}, TP: {cm[1][1]}]]")

    # --- Plotting ---
    plt.figure(figsize=(14, 6))

    # Plot 1: Score Distribution
    plt.subplot(1, 2, 1)
    plt.hist(y_scores[y_true==0], bins=50, alpha=0.7, label='Real Scores', color='blue')
    plt.hist(y_scores[y_true==1], bins=50, alpha=0.7, label='Fake Scores', color='red')
    plt.axvline(low_threshold, color='k', linestyle='--', label=f'Low Thr={low_threshold:.3f}')
    plt.axvline(high_threshold, color='k', linestyle='--', label=f'High Thr={high_threshold:.3f}')

    plt.title('Score Distribution')
    plt.xlabel('Entropy Score (bpp)')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_filename = os.path.join(ARTIFACTS_DIR, "evaluation_plots.png")
    plt.savefig(plot_filename)
    print(f"\nSaved plots to {plot_filename}")
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZED Model Evaluation Script")
    parser.add_argument("--config_path", type=str, default=os.path.join(ARTIFACTS_DIR, "zed_config.json"), help="Path to the zed_config.json file from training.")
    parser.add_argument("--test_data_dir", type=str, default=DEFAULT_TEST_DIR, help="Directory with 'real' and 'fake' subfolders for testing.")
    parser.add_argument("--results_csv", type=str, default=os.path.join(ARTIFACTS_DIR, "evaluation_results.csv"), help="Path to save the detailed CSV results.")
    args = parser.parse_args()

    # 1. Load configuration
    if not os.path.exists(args.config_path):
        print(f"Error: Configuration file not found at {args.config_path}")
        print("Please run train.py first to calibrate the model and create the config file.")
        exit()

    with open(args.config_path, 'r') as f:
        config = json.load(f)
    print("Loaded configuration:", config)

    # 2. Load model
    model = choose_entropy_model(config['model_name'], config['quality'])

    # 3. Run evaluation
    evaluate_model(model, config, args.test_data_dir, args.results_csv)