#!/usr/bin/env python
"""
image_restoration_bsd68.py

- If input_images/ is empty, downloads BSD68 (from GitHub),
  extracts images, and copies 10 of them into input_images/.
- Runs Gaussian / Median / Bilateral denoising and Wiener deblurring.
- Computes PSNR vs. ground truth.
- Saves:
    * restored images in output_images/...
    * psnr_results.csv
    * psnr_average_bar.png (bar chart of average PSNR per method)
"""

import os
import io
import zipfile
import urllib.request
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# Make cwd = script folder (important for VS Code)
# -------------------------------------------------------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
BSD68_URL = "https://github.com/clausmichele/denoising-benchmark/archive/refs/heads/master.zip"

# ---------------------------- utils --------------------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float64) / 255.0


def save_gray(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


def psnr(original: np.ndarray, restored: np.ndarray) -> float:
    mse = np.mean((original - restored) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(1.0 / np.sqrt(mse))


# ------------------------- BSD68 download ---------------------------------


def download_and_prepare_bsd68(max_images: int = 10):
    """
    Downloads BSD68 from GitHub (if needed) and copies up to max_images
    gray images into input_images/.
    """
    ensure_dir(INPUT_DIR)

    # If there are already images, don't download
    existing = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if existing:
        print(f"[INFO] input_images already has {len(existing)} images, skipping download.")
        return

    print("[INFO] Downloading BSD68 zip from GitHub...")
    resp = urllib.request.urlopen(BSD68_URL)
    data = resp.read()
    zf = zipfile.ZipFile(io.BytesIO(data))

    # BSD68 images are under: denoising-benchmark-master/BSD68/
    bsd_paths = [n for n in zf.namelist() if "BSD68/" in n and n.lower().endswith((".png", ".jpg", ".jpeg"))]
    bsd_paths.sort()

    if not bsd_paths:
        print("[WARN] Could not find BSD68 images in zip; please add images manually.")
        return

    print(f"[INFO] Found {len(bsd_paths)} BSD68 images in zip.")
    selected = bsd_paths[:max_images]
    for p in selected:
        fname = os.path.basename(p)
        out_path = os.path.join(INPUT_DIR, fname)
        with zf.open(p) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        print(f"  -> saved {out_path}")

    print("[INFO] BSD68 subset prepared in input_images/.")


# ------------------------- degradation ------------------------------------


def add_gaussian_noise(img: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    sigma_norm = sigma / 255.0
    noise = np.random.normal(0, sigma_norm, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0)


def create_motion_kernel(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    ker = np.zeros((length, length), dtype=np.float64)
    ker[length // 2, :] = 1.0
    center = (length / 2, length / 2)
    rot = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    ker = cv2.warpAffine(ker, rot, (length, length))
    s = ker.sum()
    if s != 0:
        ker /= s
    return ker


def apply_motion_blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    blurred = cv2.filter2D(img, -1, kernel)
    return np.clip(blurred, 0.0, 1.0)


# ------------------------ adaptive filters --------------------------------


def gaussian_filter_adaptive(noisy: np.ndarray, sigma_noise: float = 20.0) -> np.ndarray:
    if sigma_noise <= 10:
        ksize, sigma = 3, 0.6
    elif sigma_noise <= 25:
        ksize, sigma = 5, 1.0
    else:
        ksize, sigma = 7, 1.3
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(noisy, (ksize, ksize), sigmaX=sigma)


def median_filter_adaptive(noisy_u8: np.ndarray, noise_density: float = 0.05) -> np.ndarray:
    if noise_density <= 0.05:
        ksize = 3
    elif noise_density <= 0.10:
        ksize = 5
    else:
        ksize = 7
    if ksize % 2 == 0:
        ksize += 1
    med_u8 = cv2.medianBlur(noisy_u8, ksize)
    return med_u8.astype(np.float64) / 255.0


def bilateral_filter_adaptive(noisy_u8: np.ndarray, sigma_noise: float = 20.0) -> np.ndarray:
    if sigma_noise <= 10:
        d, s_color, s_space = 7, 40, 40
    elif sigma_noise <= 25:
        d, s_color, s_space = 9, 75, 75
    else:
        d, s_color, s_space = 11, 100, 100
    bf_u8 = cv2.bilateralFilter(noisy_u8, d=d, sigmaColor=s_color, sigmaSpace=s_space)
    return bf_u8.astype(np.float64) / 255.0


# ---------------------- Wiener deconvolution -------------------------------


def wiener_deconvolution_improved(blurred: np.ndarray, kernel: np.ndarray, K: float = 0.005) -> np.ndarray:
    h, w = blurred.shape
    kh, kw = kernel.shape

    pad_h = h + kh
    pad_w = w + kw
    pad_img = np.zeros((pad_h, pad_w), dtype=np.float64)
    pad_img[:h, :w] = blurred

    pad_k = np.zeros_like(pad_img)
    pad_k[:kh, :kw] = kernel
    pad_k = np.roll(pad_k, -kh // 2, axis=0)
    pad_k = np.roll(pad_k, -kw // 2, axis=1)

    H = np.fft.fft2(pad_k)
    G = np.fft.fft2(pad_img)

    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    F_hat = (H_conj / denom) * G

    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat)
    f_hat = f_hat[:h, :w]
    return np.clip(f_hat, 0.0, 1.0)


# -------------------------- main processing --------------------------------


def process_all():
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)
    for sub in [
        "noisy",
        "blurred",
        "denoising/gaussian",
        "denoising/median",
        "denoising/bilateral",
        "deblurring/wiener",
    ]:
        ensure_dir(os.path.join(OUTPUT_DIR, sub))

    # Download BSD68 subset if needed
    download_and_prepare_bsd68(max_images=10)

    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]
    if not images:
        print("[ERROR] No images in input_images even after download.")
        return

    results = []  # collect PSNR data

    sigma_noise = 20.0
    kernel = create_motion_kernel(length=15, angle_deg=0.0)

    for img_name in images:
        print(f"\n[INFO] Processing {img_name}")
        base, _ = os.path.splitext(img_name)
        orig = load_gray(os.path.join(INPUT_DIR, img_name))

        # ----- Denoising -----
        noisy = add_gaussian_noise(orig, sigma=sigma_noise)
        save_gray(os.path.join(OUTPUT_DIR, "noisy", f"{base}_noisy.png"), noisy)
        noisy_u8 = (noisy * 255.0).astype(np.uint8)

        gauss = gaussian_filter_adaptive(noisy, sigma_noise=sigma_noise)
        median = median_filter_adaptive(noisy_u8, noise_density=0.05)
        bilateral = bilateral_filter_adaptive(noisy_u8, sigma_noise=sigma_noise)

        save_gray(os.path.join(OUTPUT_DIR, "denoising", "gaussian", f"{base}_gaussian.png"), gauss)
        save_gray(os.path.join(OUTPUT_DIR, "denoising", "median", f"{base}_median.png"), median)
        save_gray(os.path.join(OUTPUT_DIR, "denoising", "bilateral", f"{base}_bilateral.png"), bilateral)

        # ----- Deblurring -----
        blurred = apply_motion_blur(orig, kernel)
        save_gray(os.path.join(OUTPUT_DIR, "blurred", f"{base}_blurred.png"), blurred)
        wiener_img = wiener_deconvolution_improved(blurred, kernel, K=0.005)
        save_gray(os.path.join(OUTPUT_DIR, "deblurring", "wiener", f"{base}_wiener.png"), wiener_img)

        # ----- PSNR -----
        for method, img in [
            ("gaussian", gauss),
            ("median", median),
            ("bilateral", bilateral),
            ("wiener", wiener_img),
        ]:
            value = psnr(orig, img)
            results.append({"image": base, "method": method, "psnr": value})
            print(f"  {method:9s} PSNR: {value:.2f} dB")

    # Save CSV + chart
    save_psnr_table_and_chart(results)


def save_psnr_table_and_chart(results):
    csv_path = os.path.join(OUTPUT_DIR, "psnr_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "method", "psnr"])
        for r in results:
            writer.writerow([r["image"], r["method"], f"{r['psnr']:.3f}"])
    print(f"\n[INFO] PSNR table saved to {csv_path}")

    # average PSNR per method
    methods = {}
    counts = {}
    for r in results:
        m = r["method"]
        methods[m] = methods.get(m, 0.0) + r["psnr"]
        counts[m] = counts.get(m, 0) + 1
    avg_methods = {m: methods[m] / counts[m] for m in methods}

    labels = list(avg_methods.keys())
    values = [avg_methods[m] for m in labels]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Average PSNR (dB)")
    plt.title("Average PSNR per Method on BSD68 Subset")
    chart_path = os.path.join(OUTPUT_DIR, "psnr_average_bar.png")
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] PSNR bar chart saved to {chart_path}")


if __name__ == "__main__":
    process_all()
    print("\n[INFO] Done.")
