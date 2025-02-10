import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

def compute_loss(estimated, observed):
    """Compute L2 loss between estimated and observed images."""
    return np.sum((estimated - observed) ** 2)

def refine_psf(true_image, observed_image, psf_guess, learning_rate=0.01, max_iters=50):
    """
    Iteratively refine the PSF using convolution-based optimization.
    """
    psf = psf_guess.copy()
    loss_history = []

    for _ in tqdm(range(max_iters), desc="Refining PSF"):
        # Convolve estimated PSF with true image
        estimated_image = scipy.signal.convolve2d(true_image, psf, mode='same')

        # Compute loss
        loss = compute_loss(estimated_image, observed_image)
        loss_history.append(loss)

        # Compute gradient (convolve error with flipped true image)
        error = observed_image - estimated_image
        psf_update = scipy.signal.convolve2d(error, np.flipud(np.fliplr(true_image)), mode='same')

        # 🔹 Extract the central region of psf_update to match psf size
        center_x, center_y = psf_update.shape[0] // 2, psf_update.shape[1] // 2
        half_psf_x, half_psf_y = psf.shape[0] // 2, psf.shape[1] // 2

        psf_update_cropped = psf_update[
            center_x - half_psf_x : center_x + half_psf_x + 1,
            center_y - half_psf_y : center_y + half_psf_y + 1
        ]

        # Ensure exact shape match
        psf_update = psf_update_cropped[:psf.shape[0], :psf.shape[1]]

        # Update PSF
        psf += learning_rate * psf_update

        # Ensure PSF is non-negative and normalized
        psf = np.clip(psf, 0, None)
        psf /= np.sum(psf)

    return psf, loss_history

# # Example usage
# true_image = np.random.rand(50, 50)  # Replace with actual true image
# observed_image = np.random.rand(50, 50)  # Replace with actual observed image
# psf_guess = np.ones((5, 5)) / 25  # Initial uniform guess

# refined_psf, loss_history = refine_psf(true_image, observed_image, psf_guess, max_iters=200)

# # Plot loss function history
# plt.plot(loss_history)
# plt.xlabel("Iteration")
# plt.ylabel("Loss")
# plt.title("PSF Refinement Loss")
# plt.show()
