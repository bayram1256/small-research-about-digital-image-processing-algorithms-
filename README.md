# Image Restoration Using Classical Digital Image Processing Techniques

## Abstract
This project implements and evaluates classical image restoration methods for denoising and deblurring degraded images. Gaussian, median, and bilateral filtering, along with Wiener deconvolution, are applied and compared using PSNR to analyze restoration performance and trade-offs.

---

## Project Overview
Digital images are often degraded during acquisition, transmission, or storage due to sensor noise, motion blur, or environmental conditions. Image restoration aims to recover high-quality images from such degraded observations.  
This project focuses on implementing and analyzing widely used **classical restoration techniques** in Digital Image Processing (DIP).

---

## Implemented Methods

### Denoising
- **Gaussian Filtering** – Linear smoothing for Gaussian noise reduction  
- **Median Filtering** – Non-linear filter robust to impulse noise  
- **Bilateral Filtering** – Edge-preserving smoothing using spatial and intensity similarity  

### Deblurring
- **Wiener Deconvolution** – Frequency-domain restoration using a known motion blur kernel  

---

## Workflow
1. Clean grayscale images are used as ground truth.
2. Images are synthetically degraded using:
   - Additive Gaussian noise
   - Linear motion blur
3. Restoration algorithms are applied.
4. Restoration quality is evaluated using **Peak Signal-to-Noise Ratio (PSNR)**.
5. Results are saved as restored images, numerical tables, and comparison charts.

---

## Requirements
- Python 3.9+
- NumPy
- OpenCV
- Matplotlib
- python-docx

Purpose

This project was developed as part of a Digital Image Processing course to provide hands-on experience with classical image restoration techniques and to analyze their performance under different degradation conditions.

Author
Bayram Ali
Department of Computer Engineering
Antalya Bilim University
