import numpy as np
from collections import deque
from utils.simulation_core import Detector
from sklearn.random_projection import SparseRandomProjection


class ERX(Detector):
    """
    A fast RX variant that updates the mean and covariance using exponentially moving statistics
    Implementation published in:
    Garske, Samuel and Evans, Bradley and Artlett, Christopher and Wong, KC, 2024.
    ERX - a Fast Real-Time Anomaly Detection Algorithm for Hyperspectral Line-Scanning
    arXiv preprint arXiv:2408.14947
    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int = 99,
                 n_projdims: int = 5,
                 momentum: float = 0.1,
                 normalise_md: bool = False):
        """
        :param n_bands: number of hyperspectral bands/channels from camera
        :param n_pixels: number of pixels per line (line width)
        :param buffer_len: number of lines processed before beginning detection
        :param n_projdims: number of dimensions after dimensionality reduction using random sparse projection
        :param momentum: the weighting factor placed on the most recent line's mean/covariance
                1 - momentum is the weighting placed on the previous mean/covariance
        :param normalise_md: normalises mahalanobis distance across each line
        """
        # Store inputs as attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.momentum = momentum
        self.n_projdims = n_projdims
        self.buffer_len = buffer_len
        self.normalise_md = normalise_md

        # Instantiate online tracking variables
        self.cov = None
        self.mean = None
        self.line_count = 0

        # Instantiate small identity matrix and sparse random projection (SRP) object
        self.I_small = np.identity(n_projdims) * 1e-5
        self.random_projector = SparseRandomProjection(n_components=n_projdims)
        self.random_projector.fit(X=np.ones(shape=(n_pixels, n_bands)))

    # Takes a line as input and outputs AD result
    def forward(self, x):
        """
        :param x: line of hyperspectral pixels with shape (n_pixels, n_bands)
        :return: anomaly predictions or raw detector outputs with shape (n_pixels,)
        """
        # Project hyperspectral bands to lower dimensions
        z = self.random_projector.transform(x)

        # Get mean and covariance of current line
        line_mean = z.mean(axis=0)
        line_cov = np.cov(z, rowvar=False) + self.I_small

        # Update running statistics
        if self.line_count == 0:
            # Initial values approximated by first line stats
            self.mean = line_mean
            self.cov = line_cov
        else:
            # Exponential moving stats
            self.mean -= (self.mean - line_mean) * self.momentum
            self.cov -= (self.cov - line_cov) * self.momentum

        self.line_count += 1

        # Check if buffer filled (begin anomaly detection)
        if self.line_count >= self.buffer_len:
            # Use Cholesky decomposition to get Mahalanobis distance
            delta = z - self.mean
            md = mahalanobis_cholesky(delta=delta,
                                      cov=self.cov,
                                      I_small=self.I_small)

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

        # Returns None if buffer not filled yet
        else:
            return None

    # Reset detector
    def reset(self):
        """
        Resets ERX to initial params
        """
        self.cov = None
        self.mean = None
        self.line_count = 0
        self.random_projector = SparseRandomProjection(n_components=self.n_projdims)
        self.random_projector.fit(X=np.ones(shape=(self.n_pixels, self.n_bands)))


def mahalanobis_cholesky(delta, cov, I_small):
    # Decompose covariance into lower triangle
    L = np.linalg.cholesky(cov + I_small)

    # Use forward substitution and norm to solve for Mahalanobis dist
    Z = np.linalg.solve(L, delta.T)
    return np.linalg.norm(Z, axis=0)
