import os
import argparse
import json
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score, confusion_matrix
import warnings

# Suppress harmless warnings from matplotlib
warnings.filterwarnings("ignore", module="matplotlib")

# --- Configuration ---
# Get the directory where the current script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Make your data directories relative to the script's location
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
DEFAULT_TEST_DIR = os.path.join(SCRIPT_DIR,"data/val") # Use validation set for evaluation by default
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

def zed_score_from_image(image: Image.Image, model, multiscale=(1.0, 0.75, 0.5)) -> float:
    """Computes a ZED-style surprisal score from a PIL Image object."""
    scores = []
    base_image = image.convert('RGB')
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
    return np.mean(scores)

# --- Adversarial Test Section ---

def apply_adversarial_transform(image: Image.Image) -> Image.Image:
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
    # return Image.open(buffer)

    # --- Example 2: Resizing ---
    # original_size = image.size
    # downscaled_size = (original_size[0] // 2, original_size[1] // 2)
    # image = image.resize(downscaled_size, Image.NEAREST)
    # image = image.resize(original_size, Image.LANCZOS) # Upscale back
    # return image
    
    # --- Example 3: Adding Gaussian Noise ---
    # img_np = np.array(image).astype(np.float32)
    # noise = np.random.normal(0, 15, img_np.shape) # std dev = 15
    # noisy_img_np = np.clip(img_np + noise, 0, 255).astype(np.uint8)
    # return Image.fromarray(noisy_img_np)

    # Default: No transformation
    return image


# --- Evaluation Logic ---

def evaluate_model(model, threshold, test_data_dir, results_csv):
    """Runs evaluation on the test set and saves scores and metrics."""
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
                img = Image.open(path)
                
                # Apply any defined transformations for robustness testing
                transformed_img = apply_adversarial_transform(img)

                score = zed_score_from_image(transformed_img, model)
                prediction = 1 if score > threshold else 0
                
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
    y_pred = (y_scores > threshold).astype(int)

    # --- Calculate and Display Metrics ---
    print("\n--- Evaluation Metrics ---")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Accuracy @ threshold={threshold:.4f}: {acc:.4f}")
    print(f"F1 Score @ threshold={threshold:.4f}: {f1:.4f}")
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
    plt.axvline(threshold, color='k', linestyle='--', label=f'Threshold={threshold:.3f}')
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
    evaluate_model(model, config['threshold_bpp'], args.test_data_dir, args.results_csv)