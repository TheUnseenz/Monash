"""
preprocess_images.py

Creates multiple preprocessed copies of images in an input folder.

Usage:
    python preprocess_images.py --input_dir images_raw --out_dir preprocessed

This will create:
  preprocessed/vaeround/...    (SD VAE encode-decode)
  preprocessed/jpg85/...       (JPEG compression q=85)
  preprocessed/resize512/...   (resize -> bicubic)
  preprocessed/sharpen/...     (sharpen then resize)
  preprocessed/histmatch/...   (histogram equalization in LAB)

You can then run Aeroblade on each of those folders (point Aeroblade's example set / input folder).
"""

import os
import argparse
from PIL import Image, ImageFilter, ImageOps
import numpy as np

# Optional heavy: VAE roundtrip using diffusers
USE_VAE = True

def list_images(input_dir):
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                yield os.path.join(root, f)

def save_img(img: Image.Image, src_path, out_root, transform_name, input_root):
    rel = os.path.relpath(src_path, start=input_root)   # key fix
    dest_dir = os.path.join(out_root, transform_name, os.path.dirname(rel))
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, os.path.basename(rel))
    img.save(dest_path, quality=95)
    return dest_path


def resize_image(img, size=(1024,1024)):
    return img.resize(size, resample=Image.BICUBIC)

def jpeg_compress(img, quality=85):
    # return a new PIL image that simulates JPEG compression loss
    from io import BytesIO
    buff = BytesIO()
    img.save(buff, format='JPEG', quality=quality)
    buff.seek(0)
    return Image.open(buff).convert('RGB')

def sharpen_image(img):
    # Example: sharpen + slight unsharp mask
    img2 = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
    return img2

def hist_equalize_lab(img):
    # Convert to LAB, equalize L channel
    import cv2
    arr = np.array(img.convert('RGB'))[:, :, ::-1]  # RGB->BGR for cv2
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    rgb = bgr[:, :, ::-1]
    return Image.fromarray(rgb)

# --- Optional: SD VAE roundtrip (requires diffusers + torch)
def vae_roundtrip_pil(img: Image.Image, vae=None, device="cpu"):
    """
    Convert PIL image -> latent via SD VAE encoder -> decode back.
    vae should be an AutoencoderKL or compatible model from diffusers.
    Works at 1024 resolution expected; will resize image to 1024x1024.
    """
    if vae is None:
        raise RuntimeError("VAE model not provided")
    import torch
    from torchvision import transforms
    img = img.convert("RGB")
    img_resized = img.resize((1024, 1024), resample=Image.BICUBIC)
    to_tensor = transforms.ToTensor()
    x = to_tensor(img_resized).unsqueeze(0).to(device) * 2.0 - 1.0  # [-1,1]
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample() * vae.scaling_factor
        decoded = vae.decode(latent / vae.scaling_factor).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    decoded = (decoded[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
    return Image.fromarray(decoded)

def main(args):
    inp = args.input_dir
    out = args.out_dir
    # os.makedirs(out, exist_ok=True)

    # Optionally load SD VAE
    vae = None
    if args.enable_vae:
        try:
            from diffusers import AutoencoderKL
            import torch
            # choose a VAE checkpoint compatible with SD. Adjust repo/model as needed.
            vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(args.device)
            vae.eval()
            print("Loaded VAE.")
        except Exception as e:
            print("Failed to load VAE (skipping). Exception:", e)
            vae = None

    for path in list_images(inp):
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print("skip", path, e)
            continue

        # base resize (always produce 512 copy)
        r = resize_image(img, (1024, 1024))
        save_img(r, path, out, "resize1024", inp)

        # jpeg compressed
        j = jpeg_compress(r, quality=85)
        save_img(j, path, out, "jpg85", inp)

        # sharpen
        s = sharpen_image(r)
        save_img(s, path, out, "sharpen", inp)

        # histogram equalize (LAB)
        try:
            he = hist_equalize_lab(r)
            save_img(he, path, out, "histlab", inp)
        except Exception as e:
            print("histlab failed", e)

        # VAE roundtrip
        if args.enable_vae and vae is not None:
            try:
                vr = vae_roundtrip_pil(img, vae=vae, device=args.device)
                save_img(vr, path, out, "vaeround", inp)
            except Exception as e:
                print("vae roundtrip failed for", path, e)

    print("Done. Preprocessed folders in:", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="Sem3/FIT5230MaliciousAI/Assignments/cloned_repos/aeroblade/real_img/real_1024")
    parser.add_argument("--out_dir", default="Sem3/FIT5230MaliciousAI/Assignments/cloned_repos/aeroblade/real_img/real_1024_processed")
    parser.add_argument("--enable_vae", default=True, action="store_true", help="Use SD VAE roundtrip (heavy, needs diffusers + torch)")
    parser.add_argument("--device", default="cuda", help="device for VAE (cpu or cuda)")
    args = parser.parse_args()
    main(args)
