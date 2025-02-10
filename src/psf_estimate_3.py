import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

def pad_psf(psf, shape):
    """Pad PSF to match image size with centered placement."""
    padded_psf = np.zeros(shape)
    psf_shape = psf.shape
    insert_x = (shape[0] - psf_shape[0]) // 2
    insert_y = (shape[1] - psf_shape[1]) // 2
    padded_psf[insert_x:insert_x + psf_shape[0], insert_y:insert_y + psf_shape[1]] = psf
    return padded_psf

def fft_convolve(image, psf):
    """Efficient FFT-based convolution."""
    return scipy.signal.fftconvolve(image, psf, mode='same')

def wiener_deconvolution(observed, original, reg=1e-3):
    """Estimate PSF using Wiener deconvolution in the frequency domain."""
    O = np.fft.fft2(observed)
    G = np.fft.fft2(original)
    H = O * np.conj(G) / (np.abs(G) ** 2 + reg)
    psf_est = np.real(np.fft.ifft2(H))
    psf_est -= psf_est.min()
    psf_est /= psf_est.sum()
    return psf_est

def estimate_psf(observed, original, init_psf, iterations=500, alpha=0.1, reg=1e-3):
    """Iteratively refine PSF estimation using Wiener deconvolution and track loss."""
    psf = pad_psf(init_psf, observed.shape)  # Ensure PSF is full image size
    loss_history = []

    for i in range(iterations):
        # Convolve original image with current PSF estimate
        estimated_image = fft_convolve(original, psf)

        # Compute residual error
        error = observed - estimated_image
        loss = np.linalg.norm(error)  # L2 norm of error

        # Store loss for tracking
        loss_history.append(loss)

        # Wiener deconvolution step
        psf_update = wiener_deconvolution(observed, estimated_image, reg)
        psf = (1 - alpha) * psf + alpha * psf_update  # Weighted update

        # Normalize PSF
        psf -= psf.min()
        psf /= psf.sum()

        print(f"Iteration {i+1}: Loss = {loss}")

    return psf, loss_history

