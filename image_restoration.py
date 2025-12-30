import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

# ================== CONFIG ==================

# Folder with original images (relative to this script)
INPUT_DIR = "input_images"

# Root folder for all outputs
OUTPUT_DIR = "output_images"


# ================== UTIL FUNCTIONS ==================

def ensure_dir(path: str) -> None:
    """Create folder if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_gray(path: str) -> np.ndarray:
    """Load image as grayscale float64 in [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img.astype(np.float64) / 255.0


def save_gray(path: str, img: np.ndarray) -> None:
    """Save grayscale float image [0,1] as uint8 PNG/JPG."""
    ensure_dir(os.path.dirname(path))
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(path, img_uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """
    Add Gaussian noise (sigma in 0–255 range).
    """
    sigma_norm = sigma / 255.0
    noise = np.random.normal(0, sigma_norm, img.shape)
    noisy = img + noise
    return np.clip(noisy, 0.0, 1.0)


def create_motion_kernel(length: int = 15, angle_deg: float = 0.0) -> np.ndarray:
    """
    Create a simple motion blur kernel: a straight line of given length and angle.
    """
    kernel = np.zeros((length, length), dtype=np.float64)
    kernel[length // 2, :] = 1.0  # horizontal line in the middle

    center = (length / 2, length / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot_mat, (length, length))

    s = kernel.sum()
    if s != 0:
        kernel /= s
    return kernel


def apply_motion_blur(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apply motion blur using 2D convolution via OpenCV.
    """
    blurred = cv2.filter2D(img, -1, kernel)
    return np.clip(blurred, 0.0, 1.0)


def wiener_deconvolution(blurred: np.ndarray,
                         kernel: np.ndarray,
                         K: float = 0.01) -> np.ndarray:
    """
    Simple Wiener deconvolution implemented using FFT.
    blurred, kernel: float64 in [0,1].
    K: noise-to-signal ratio (regularization).
    """
    h, w = blurred.shape
    kh, kw = kernel.shape

    # pad kernel to image size
    pad = np.zeros_like(blurred)
    pad[:kh, :kw] = kernel

    # shift kernel to center
    pad = np.roll(pad, -kh // 2, axis=0)
    pad = np.roll(pad, -kw // 2, axis=1)

    H = np.fft.fft2(pad)
    G = np.fft.fft2(blurred)

    H_conj = np.conj(H)
    denom = (H * H_conj) + K
    F_hat = (H_conj / denom) * G

    f_hat = np.fft.ifft2(F_hat)
    f_hat = np.real(f_hat)
    return np.clip(f_hat, 0.0, 1.0)


# ================== MAIN PROCESSING ==================

def process_image(img_name: str):
    print(f"\n[INFO] Processing: {img_name}")
    base, _ = os.path.splitext(img_name)
    img_path = os.path.join(INPUT_DIR, img_name)

    # 1) Load original
    orig = load_gray(img_path)

    # ================== DENOISING PIPELINE ==================

    # create noisy version
    noisy = add_gaussian_noise(orig, sigma=20.0)
    save_gray(os.path.join(OUTPUT_DIR, "noisy", f"{base}_noisy.png"), noisy)

    # Gaussian filter (denoising)
    gauss = cv2.GaussianBlur(noisy, (5, 5), 1.0)
    save_gray(
        os.path.join(OUTPUT_DIR, "denoising", "gaussian", f"{base}_gaussian.png"),
        gauss,
    )

    # Median filter (OpenCV works on uint8)
    noisy_u8 = (noisy * 255.0).astype(np.uint8)
    median_u8 = cv2.medianBlur(noisy_u8, 5)
    median = median_u8.astype(np.float64) / 255.0
    save_gray(
        os.path.join(OUTPUT_DIR, "denoising", "median", f"{base}_median.png"),
        median,
    )

    # Bilateral filter
    bilateral_u8 = cv2.bilateralFilter(noisy_u8, d=9, sigmaColor=75, sigmaSpace=75)
    bilateral = bilateral_u8.astype(np.float64) / 255.0
    save_gray(
        os.path.join(OUTPUT_DIR, "denoising", "bilateral", f"{base}_bilateral.png"),
        bilateral,
    )

    # ================== DEBLURRING PIPELINE ==================

    # create blurred version with motion kernel
    kernel = create_motion_kernel(length=15, angle_deg=0.0)
    blurred = apply_motion_blur(orig, kernel)
    save_gray(os.path.join(OUTPUT_DIR, "blurred", f"{base}_blurred.png"), blurred)

    # Wiener deconvolution
    wiener_restored = wiener_deconvolution(blurred, kernel, K=0.01)
    save_gray(
        os.path.join(OUTPUT_DIR, "deblurring", "wiener", f"{base}_wiener.png"),
        wiener_restored,
    )

    print(f"[DONE] Saved results for {img_name}")


def main():
    # Ensure base folders
    ensure_dir(INPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    # Subfolders for better structure
    ensure_dir(os.path.join(OUTPUT_DIR, "noisy"))
    ensure_dir(os.path.join(OUTPUT_DIR, "blurred"))
    ensure_dir(os.path.join(OUTPUT_DIR, "denoising", "gaussian"))
    ensure_dir(os.path.join(OUTPUT_DIR, "denoising", "median"))
    ensure_dir(os.path.join(OUTPUT_DIR, "denoising", "bilateral"))
    ensure_dir(os.path.join(OUTPUT_DIR, "deblurring", "wiener"))

    # List input images
    images = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    if not images:
        print(f"[ERROR] No images found in '{INPUT_DIR}'.")
        print("→ Put some .png/.jpg images into that folder and run again.")
        return

    for img_name in images:
        process_image(img_name)

    print("\n[INFO] All images processed.")
    print(f"[INFO] Check output folders under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
