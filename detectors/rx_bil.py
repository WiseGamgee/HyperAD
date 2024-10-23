import numpy as np
from collections import deque
from scipy.spatial import distance
from utils.simulation_core import Detector


class RX_BIL(Detector):
    """
    Implementation based on:
    Du, Q. and Nekovei, R., 2009.
    Fast real-time onboard processing of hyperspectral imagery for detection and classification.
    Journal of Real-Time Image Processing, 4(3), pp.273-286.
    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 pixel_dropout: float = None,
                 line_offset: int = 0,
                 normalise_md: bool = False):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param pixel_dropout: fraction of pixels to be dropped out per line
        :param line_offset: no. of lines between current line and the line prediction is made for
               e.g. line_offset = 1 will predict anomalies for the second last line captured
        :param normalise_md: normalises mahalanobis distance across each line
        """

        # Store attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.pixel_dropout = pixel_dropout
        self.normalise_md = normalise_md

        # Create buffer
        self.buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.line_count = 0
        self.pixel_count = 0
        self.inv_corr = None
        self.n_pixels_dropout = int((1 - pixel_dropout) * n_pixels)
        self.I = np.identity(self.n_pixels_dropout)
        self.zero_mean = np.zeros((1, n_bands))

    def forward(self, x):
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Drop pixels from array if specified
        if self.pixel_dropout:
            indices = np.random.choice(np.arange(self.n_pixels),
                                       size=self.n_pixels_dropout,
                                       replace=False)
            x = x[indices]

        # Use first line inverse corr as initial value
        if self.line_count == 1:
            self.inv_corr = np.linalg.inv(x.T @ x)
        else:
            # Use Woodbury to update inverse corr line by line
            self.inv_corr = inv_corr_update_line(inv_corr=self.inv_corr,
                                                 x=x,
                                                 I=self.I)

        # Check if buffer filled
        if self.line_count >= self.buffer_len:
            # Get mahalanobis distance of specified line
            md = distance.cdist(self.buffer[-self.line_offset - 1],
                                self.zero_mean,
                                metric='mahalanobis',
                                VI=self.inv_corr).flatten()

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

            # Returns None if buffer not filled yet
        else:
            return None

    def reset(self):
        self.inv_corr = None
        self.line_count = 0
        self.pixel_count = 0
        self.buffer = deque(maxlen=self.buffer_len)


def inv_corr_update_line(inv_corr, x, I):
    a = inv_corr @ x.T  # n_bands * n_pixels
    b = np.linalg.inv(I + (x @ a))  # n_pixels * n_pixels
    c = x @ inv_corr  # n_pixels * n_bands
    new_inv_corr = inv_corr - a @ b @ c
    return new_inv_corr


def inv_cov_update_line(inv_cov, x, mean, n):
    # Temp variables for simplifying calculation
    Z = (x - mean).T
    A_inv = (n / (n - 1)) * inv_cov
    U = (1 / np.sqrt(n)) * Z
    V_t = U.T

    a = A_inv @ U
    I = np.identity(len(x))
    b = np.linalg.inv(I + (V_t @ a))
    c = V_t @ A_inv

    # Calculate updated covariance
    new_inv_cov = A_inv - (a @ b @ c)
    return new_inv_cov
