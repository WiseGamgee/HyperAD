import numpy as np
from numba import njit
from collections import deque
from scipy.spatial import distance
from utils.simulation_core import Detector


class LblAD(Detector):
    """
    Implementation based on:
    Horstrand, P., Díaz, M., Guerra, R., López, S. and López, J.F., 2019.
    A novel hyperspectral anomaly detection algorithm for real-time applications with push-broom sensors.
    IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(12), pp.4787-4797.
    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 pca_dims: int = 3,
                 line_offset: int = 0,
                 normalise_md: bool = False):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param pca_dims: no. of bands after dimensionality reduction
        :param normalise_md: normalises mahalanobis distance across each line
        """

        # Store inputs as attributes
        self.n_bands = n_bands
        self.pca_dims = pca_dims
        self.n_pixels = n_pixels
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.normalise_md = normalise_md

        # Create buffer
        self.buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.md_std = 1
        self.md_mean = 0
        self.line_count = 0
        self.pixel_count = 0
        self.mean = np.zeros(n_bands)
        self.sum_x = np.zeros(n_bands)
        self.zero_mean = np.zeros((1, pca_dims))
        self.cov_prime = np.zeros((n_bands, n_bands))

    def forward(self, x):
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Update tracked variables with new line information
        self.sum_x += x.sum(axis=0)
        self.pixel_count += self.n_pixels
        self.mean = self.sum_x / self.pixel_count
        cov_update = ((x - self.mean).T @ (x - self.mean))
        cov_prime = self.cov_prime + cov_update

        # Check if buffer filled
        if self.line_count >= self.buffer_len:
            # Calculate covariance
            cov = cov_prime / self.pixel_count

            # Get eigenvalues and eigenvectors of covariance matrix
            rand_vec = np.random.rand(len(cov))
            E_0 = rand_vec / rand_vec.sum()
            e_arr, E_arr = subspace_calculation(cov_prime=cov,
                                                E_0=E_0,
                                                pca_dims=self.pca_dims)

            # Project line into new subspace
            Y = E_arr @ (self.buffer[-self.line_offset - 1] - self.mean).T
            cov_transformed = np.diag(e_arr)

            # Get mahalanobis distance
            md = distance.cdist(Y.T,
                                self.zero_mean,
                                metric='mahalanobis',
                                VI=np.linalg.inv(cov_transformed)).flatten()

            # Update the Mahalanobis distance mean and standard deviation only using background pixels
            background_mask = np.where(md > self.md_mean + self.md_std * 15, 0, 1)
            n_background = background_mask.sum()

            self.md_mean = (md * background_mask).sum() / n_background
            self.md_std = (((md - self.md_mean) ** 2) * background_mask).sum() / (n_background - 1)

            # If an anomaly is detected don't update cov prime
            if n_background == len(background_mask):
                self.cov_prime = cov_prime

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

        else:
            # Update covariance and return None if buffer not filled yet
            self.cov_prime = cov_prime
            return None

    def reset(self):
        self.buffer = deque(maxlen=self.buffer_len)
        self.md_std = 1
        self.md_mean = 0
        self.line_count = 0
        self.pixel_count = 0
        self.mean = np.zeros(self.n_bands)
        self.sum_x = np.zeros(self.n_bands)
        self.zero_mean = np.zeros((1, self.pca_dims))
        self.cov_prime = np.zeros((self.n_bands, self.n_bands))


def subspace_calculation(cov_prime, E_0, pca_dims, tol=1e-5):
    # Get initial eigenvalues and eigenvectors via power method
    e_0, E_0 = power_method(cov_prime, E_0, tol)

    # Instantiate tracking variables
    i = 2
    B = cov_prime
    e_arr, E_arr = [e_0], [E_0]

    # Loop through number of PCs required
    while i <= pca_dims:
        # Deflate covariance matrix
        B -= (E_0 @ E_0)

        # Get updated eigenvalues and eigenvectors
        e_new, E_new = power_method(B, E_0, tol)
        e_arr.append(e_new)
        E_arr.append(E_new)

        # Increment to next dim
        i += 1
    # Return all eigenvalues and vectors
    return np.array(e_arr), np.array(E_arr)


@njit
def power_method(cov_prime, E_0, tol=1e-5):
    # Initialise eigenvalues
    e_0 = 0
    e_1 = (E_0 @ cov_prime @ E_0) / (E_0 @ E_0)
    E_1 = (cov_prime @ E_0) / np.linalg.norm(cov_prime @ E_0)

    # Iterate until Eigenvalue convergence
    while np.abs(e_0 - e_1) > tol:

        # Calculate updated eigenvalue
        e_1 = (E_0 @ cov_prime @ E_0) / (E_0 @ E_0)

        # Calculate updated eigenvector
        E_1 = (cov_prime @ E_0) / np.linalg.norm(cov_prime @ E_0)

        # Update previous eigenvalue and eigenvector
        e_0 = e_1
        E_0 = E_1
    return e_1, E_1
