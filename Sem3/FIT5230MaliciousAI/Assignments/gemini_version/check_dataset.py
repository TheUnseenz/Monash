import os
import PIL.Image as pil_image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def check_image_size_distributions(data_dir: str, required_factor: int = 64):
    """
    Checks the distribution of image sizes and padding needed for real and fake images.
    
    Args:
        data_dir (str): The path to the main data directory (e.g., 'data/train').
        required_factor (int): The padding factor used by the model.
    """
    real_dir = os.path.join(data_dir, "real")
    fake_dir = os.path.join(data_dir, "fake")
    print(real_dir)

    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print("Error: 'real' and 'fake' subdirectories not found.")
        return

    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
    fake_paths = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
    
    # Take a random sample of images to speed up the analysis
    sample_size = 5000 
    real_paths_sample = np.random.choice(real_paths, min(len(real_paths), sample_size), replace=False)
    fake_paths_sample = np.random.choice(fake_paths, min(len(fake_paths), sample_size), replace=False)

    def analyze_sizes(paths):
        widths, heights, pad_amounts = [], [], []
        for path in tqdm(paths, desc="Analyzing images"):
            try:
                with pil_image.open(path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    
                    # Calculate padding needed for both dimensions
                    pad_w = required_factor - (w % required_factor) if w % required_factor != 0 else 0
                    pad_h = required_factor - (h % required_factor) if h % required_factor != 0 else 0
                    pad_amounts.append(pad_w + pad_h)
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")
                continue
        return np.array(widths), np.array(heights), np.array(pad_amounts)

    print("Analyzing Real Images...")
    real_widths, real_heights, real_pads = analyze_sizes(real_paths_sample)
    
    print("Analyzing Fake Images...")
    fake_widths, fake_heights, fake_pads = analyze_sizes(fake_paths_sample)

    # Plotting the distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Image Size and Padding Analysis', fontsize=16)

    # Plot 1: Width Distribution
    axes[0, 0].hist(real_widths, bins=50, alpha=0.7, label='Real', color='blue')
    axes[0, 0].hist(fake_widths, bins=50, alpha=0.7, label='Fake', color='red')
    axes[0, 0].set_title('Width Distribution')
    axes[0, 0].set_xlabel('Width (pixels)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()

    # Plot 2: Height Distribution
    axes[0, 1].hist(real_heights, bins=50, alpha=0.7, label='Real', color='blue')
    axes[0, 1].hist(fake_heights, bins=50, alpha=0.7, label='Fake', color='red')
    axes[0, 1].set_title('Height Distribution')
    axes[0, 1].set_xlabel('Height (pixels)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    
    # Plot 3: Padding Needed (Width)
    axes[1, 0].hist(real_pads[real_pads > 0], bins=required_factor, range=(1, required_factor), alpha=0.7, label='Real', color='blue')
    axes[1, 0].hist(fake_pads[fake_pads > 0], bins=required_factor, range=(1, required_factor), alpha=0.7, label='Fake', color='red')
    axes[1, 0].set_title(f'Total Padding Needed (pixels, multiple of {required_factor})')
    axes[1, 0].set_xlabel('Pixels of Padding')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # Plot 4: Box Plot of Padding
    axes[1, 1].boxplot([real_pads, fake_pads], labels=['Real', 'Fake'])
    axes[1, 1].set_title('Distribution of Total Padding')
    axes[1, 1].set_ylabel('Total Padding (pixels)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Example usage (assuming 'data/train' is where your images are saved)
# You should run this after your main script has downloaded and saved the data.
check_image_size_distributions('C:\PersonalStuff\Monash\Sem3\FIT5230MaliciousAI\Assignments\data\\train')