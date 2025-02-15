import numpy as np
import scipy.signal
import scipy.fft
import matplotlib.pyplot as plt
from tqdm import tqdm

# Try to import CuPy (GPU acceleration)
try:
    import cupy as cp
    GPU_AVAILABLE = True
    xp = cp  # Use CuPy for GPU computations
    print("Using GPU acceleration with CuPy!")

    def fft_convolve_gpu(image, kernel):
        """Perform FFT-based convolution using CuPy (GPU)."""
        img_freq = cp.fft.fft2(image)
        kernel_freq = cp.fft.fft2(kernel, s=image.shape)  # Match shape
        convolved = cp.fft.ifft2(img_freq * kernel_freq).real  # Multiply in frequency domain
        return convolved

    fftconvolve = fft_convolve_gpu  # Use GPU-based FFT convolution

except ImportError:
    GPU_AVAILABLE = False
    xp = np  # Use NumPy for CPU computations
    fftconvolve = scipy.signal.fftconvolve
    print("Using CPU (install CuPy for GPU acceleration)")

def compute_loss(estimated, observed):
    """Compute L2 loss between estimated and observed images."""
    return xp.sum((estimated - observed) ** 2)

def refine_psf(true_image, observed_image, psf_guess, learning_rate=0.01, reg=1e-4, momentum=0.1, max_iters=1000):
    """
    Iteratively refine the PSF using an optimized approach with FFT-based convolution.
    Uses CuPy if GPU is available.
    """
    # Move data to GPU if available
    true_image = xp.asarray(true_image)
    observed_image = xp.asarray(observed_image)
    psf = xp.asarray(psf_guess)
    
    # Track previous update (for momentum)
    prev_update = xp.zeros_like(psf)
    
    loss_history = []

    for _ in tqdm(range(max_iters), desc="Refining PSF"):
        # Compute estimated image via FFT convolution
        estimated_image = fftconvolve(true_image, psf)

        # Compute loss
        loss = compute_loss(estimated_image, observed_image)
        loss_history.append(loss)

        # Compute gradient (error convolved with flipped true image)
        error = observed_image - estimated_image
        # Compute PSF update using error
        psf_update = fftconvolve(error, xp.flipud(xp.fliplr(true_image)))

        # Crop the PSF update to match PSF size
        center_x, center_y = psf_update.shape[0] // 2, psf_update.shape[1] // 2
        half_psf_x, half_psf_y = psf.shape[0] // 2, psf.shape[1] // 2

        psf_update = psf_update[
            center_x - half_psf_x : center_x + half_psf_x + 1,
            center_y - half_psf_y : center_y + half_psf_y + 1
        ]

        # Ensure exact shape match
        psf_update = psf_update[:psf.shape[0], :psf.shape[1]]

        # Regularization
        psf_update /= (xp.abs(psf_update).max() + reg)

        # Apply momentum
        prev_update = xp.zeros_like(psf)  # Ensure `prev_update` matches PSF size
        psf_update = momentum * prev_update + (1 - momentum) * psf_update
        prev_update = psf_update  # Store for next iteration

        # Update PSF
        psf += learning_rate * psf_update

        # Ensure PSF is non-negative and normalized
        psf = xp.clip(psf, 0, None)
        psf /= xp.sum(psf)

    # Move back to CPU if using GPU
    return xp.asnumpy(psf), loss_history if GPU_AVAILABLE else (psf, loss_history)

