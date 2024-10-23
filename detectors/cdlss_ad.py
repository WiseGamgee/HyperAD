import numpy as np
from collections import deque
from utils.simulation_core import Detector


class CDLSS_AD(Detector):
    """
    Implementation based on:
    Zhang, L., Peng, B., Zhang, F., Wang, L., Zhang, H., Zhang, P. and Tong, Q., 2017.
    Fast real-time causal linewise progressive hyperspectral anomaly detection via cholesky decomposition.
    IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 10(10), pp.4614-4629.
    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 line_offset: int = 0,
                 normalise_md: bool = False):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param line_offset: no. of lines between current line and the line prediction is made for
                e.g. line_offset = 1 will predict anomalies for the second last line captured
        :param normalise_md: normalises mahalanobis distance across each line
        """

        # Store inputs as attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.normalise_md = normalise_md

        # Create buffers (first for raw lines, second for autocorrelations)
        self.buffer = deque(maxlen=buffer_len)
        self.corr_buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.line_count = 0
        self.corr = np.zeros((n_bands, n_bands))
        self.mini_I = np.identity(n_bands) * 1e-6

    # Purpose of forward is to take line input and output AD result
    def forward(self, x):
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Calculate autocorrelation of current line
        line_corr = (x.T @ x) / self.n_pixels

        # Check if buffer not yet filled
        if self.line_count < self.buffer_len:
            # Store current line correlation in buffer
            self.corr_buffer.append(line_corr)

            # Updates autocorrelation
            a = (self.line_count - 1) / self.line_count
            b = 1 / self.line_count
            self.corr = a * self.corr + b * line_corr

            # Returns none since buffer not filled yet
            return None

        else:
            # Get mahalnobis distance for current line using previous corr
            md = mahalanobis_cholesky(delta=self.buffer[-self.line_offset - 1],
                                      cov=self.corr,
                                      I_mini=self.mini_I)

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()

            # Update autocorrelation and return predictions
            old_corr = self.corr_buffer[0]
            self.corr_buffer.append(line_corr)
            self.corr += (1 / self.buffer_len) * (line_corr - old_corr)
            return md

    # Reset detector
    def reset(self):
        self.line_count = 0
        self.buffer = deque(maxlen=self.buffer_len)
        self.corr_buffer = deque(maxlen=self.buffer_len)
        self.corr = np.zeros((self.n_bands, self.n_bands))


def mahalanobis_cholesky(delta, cov, I_mini):
    # Decompose covariance into lower triangle
    L = np.linalg.cholesky(cov + I_mini)

    # Use forward substitution and norm to solve for Mahalanobis dist
    Z = np.linalg.solve(L, delta.T)
    return np.linalg.norm(Z, axis=0)
