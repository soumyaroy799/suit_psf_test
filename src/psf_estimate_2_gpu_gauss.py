import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm

def add_gaussian(psf, center, amplitude, sigma):
    """Adds a Gaussian blob to the PSF."""
    x, y = np.meshgrid(np.arange(psf.shape[1]), np.arange(psf.shape[0]))
    gauss = amplitude * np.exp(-((x - center[1])**2 + (y - center[0])**2) / (2 * sigma**2))
    return psf + gauss

def refine_psf(true_image, observed_image, psf_guess, num_gaussians=1, max_iters=1000):
    """
    Iteratively refines the PSF by adding Gaussian blobs to minimize error.
    """
    psf = psf_guess.copy()
    loss_history = []

    for _ in tqdm(range(max_iters), desc="Refining PSF"):
        # Convolve estimated PSF with true image
        estimated_image = scipy.signal.convolve2d(true_image, psf, mode='same')

        # Compute error
        error = observed_image - estimated_image
        loss = np.sum(error**2)
        loss_history.append(loss)

        # Identify error peak locations to place Gaussians
        for _ in range(num_gaussians):
            peak_idx = np.unravel_index(np.argmax(np.abs(error)), error.shape)
            peak_value = error[peak_idx]
            psf = add_gaussian(psf, peak_idx, amplitude=0.1 * peak_value, sigma=3)

        # Normalize PSF
        psf = np.clip(psf, 0, None)
        psf /= np.sum(psf)

    return psf, loss_history
