import numpy as np
import scipy.signal
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_loss(estimated, observed):
    """Compute L2 loss between estimated and observed images."""
    return np.sum((estimated - observed) ** 2)

def adam_update(psf, gradient, m, v, t, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Adam optimizer update rule."""
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)

    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    psf -= lr * m_hat / (cp.sqrt(v_hat) + epsilon)
    psf = cp.maximum(psf, 0)  # Ensure PSF is non-negative
    psf /= cp.sum(psf)  # Normalize PSF

    return psf, m, v

def refine_psf(true_image, observed_image, psf_guess, lr=0.01, max_iters=50):
    """
    Iteratively refine the PSF using Adam optimization.
    """
    try:
        xp = cp  # Try using GPU
        true_image, observed_image, psf = map(cp.asarray, (true_image, observed_image, psf_guess))
    except:
        xp = np  # Fallback to CPU
        true_image, observed_image, psf = np.asarray(true_image), np.asarray(observed_image), np.asarray(psf_guess)

    loss_history = []
    m, v = xp.zeros_like(psf), xp.zeros_like(psf)

    for t in tqdm(range(1, max_iters + 1), desc="Refining PSF with Adam"):
        # Convolve estimated PSF with true image
        estimated_image = scipy.signal.convolve2d(true_image.get(), psf.get(), mode='same')
        estimated_image = xp.asarray(estimated_image)

        # Compute loss
        loss = compute_loss(estimated_image, observed_image)
        loss_history.append(loss)

        # Compute gradient (convolve error with flipped true image)
        error = observed_image - estimated_image
        psf_update = scipy.signal.convolve2d(error.get(), xp.flipud(xp.fliplr(true_image)).get(), mode='same')
        psf_update = xp.asarray(psf_update)

        # Crop to match PSF size
        center_x, center_y = psf_update.shape[0] // 2, psf_update.shape[1] // 2
        half_psf_x, half_psf_y = psf.shape[0] // 2, psf.shape[1] // 2

        psf_update_cropped = psf_update[
            center_x - half_psf_x : center_x + half_psf_x + 1,
            center_y - half_psf_y : center_y + half_psf_y + 1
        ]

        psf_update = psf_update_cropped[:psf.shape[0], :psf.shape[1]]

        # Update PSF using Adam optimizer
        psf, m, v = adam_update(psf, psf_update, m, v, t, lr=lr)

    return psf.get() if xp == cp else psf, loss_history
