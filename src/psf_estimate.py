import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

def compute_loss(estimated, observed):
    """Compute mean squared error between estimated and observed images."""
    return np.mean((estimated - observed) ** 2)

def update_psf(psf, true_image, observed_convolved, learning_rate=0.01):
    """Adjust the PSF iteratively by minimizing the difference."""
    estimated_convolved = convolve2d(true_image, psf, mode='same')  # Simulate convolution
    error = observed_convolved - estimated_convolved  # Compute difference

    # Compute PSF gradient by convolving the error with the flipped true image
    psf_update = convolve2d(error, np.flipud(np.fliplr(true_image)), mode='same')

    # 🔹 Extract the central region of psf_update to match psf size
    center_x, center_y = psf_update.shape[0] // 2, psf_update.shape[1] // 2
    half_psf_x, half_psf_y = psf.shape[0] // 2, psf.shape[1] // 2

    psf_update_cropped = psf_update[
        center_x - half_psf_x : center_x + half_psf_x + 1,
        center_y - half_psf_y : center_y + half_psf_y + 1
    ]

    # Ensure exact shape match
    psf_update_cropped = psf_update_cropped[:psf.shape[0], :psf.shape[1]]

    # Update PSF
    new_psf = psf + learning_rate * psf_update_cropped

    # Ensure PSF is non-negative and normalized
    new_psf = np.clip(new_psf, 0, None)
    new_psf /= np.sum(new_psf)  # Normalize to preserve energy

    return new_psf

