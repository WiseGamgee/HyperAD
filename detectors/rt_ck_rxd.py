import numpy as np
from numba import njit
from collections import deque
from scipy.spatial import distance
from utils.simulation_core import Detector
from threadpoolctl import threadpool_limits


class RT_CK_RXD(Detector):
    """
    Implementation based on:
    Chen, S.Y., Wang, Y., Wu, C.C., Liu, C. and Chang, C.I., 2014.
    Real-time causal processing of anomaly detection for hyperspectral imagery.
    IEEE transactions on aerospace and electronic systems, 50(2), pp.1511-1534.
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

        # Store variables
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.normalise_md = normalise_md

        # Create buffer
        self.buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.line_count = 0
        self.pixel_count = 0
        self.mean = np.zeros(n_bands)
        self.inv_cov = np.identity(n_bands)

    # Purpose of forward is to take line input and output AD result
    def forward(self, x):
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Check if buffer filled for first time
        if self.line_count == self.buffer_len:
            # Convert line buffer into pixel array
            pixel_array = np.array(self.buffer).reshape(-1, self.n_bands)

            # Update total number of pixels processed
            self.pixel_count += len(pixel_array)

            # Calculate mean and covariance for all pixels
            self.mean = pixel_array.mean(axis=0)
            cov = np.cov(pixel_array, rowvar=False)
            self.inv_cov = np.linalg.inv(cov)

            # Get mahalanobis distance
            md = distance.cdist(self.buffer[-self.line_offset - 1],
                                self.mean.reshape(1, -1),
                                metric='mahalanobis',
                                VI=self.inv_cov).flatten()

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

        # Check if buffer is rolling
        elif self.line_count > self.buffer_len:
            # Loop through all pixels and update inverse covariance matrix
            md = []
            y = self.buffer[-self.line_offset - 1]
            for i in range(len(x)):
                # Update the total number of pixels processed
                self.pixel_count += 1

                # Extract current pixel
                x_i = x[i]
                y_i = y[i]

                # Get mean using recursive update
                self.mean += (x_i - self.mean) / self.pixel_count

                # Use Woodbury for invcov update
                self.inv_cov = inv_cov_update_pixel(inv_cov=self.inv_cov,
                                                    x_i=x_i,
                                                    mean=self.mean,
                                                    n_pixels=self.pixel_count)

                # Get Mahalanobis distance for current pixel and append to list
                md_i = distance.mahalanobis(u=y_i,
                                            v=self.mean,
                                            VI=self.inv_cov)
                md.append(md_i)

            # Convert to array
            md = np.array(md)

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

        # Returns None if buffer not filled yet
        else:
            return None

    # Reset detector
    def reset(self):
        self.line_count = 0
        self.pixel_count = 0
        self.mean = np.zeros(self.n_bands)
        self.inv_cov = np.identity(self.n_bands)
        self.buffer = deque(maxlen=self.buffer_len)


@njit
def inv_cov_update_pixel(inv_cov: np.ndarray,
                         x_i: np.ndarray,
                         mean: np.ndarray,
                         n_pixels: int):
    """
    :param inv_cov: inverse covariance matrix
    :param x_i: hyperspectral pixel vector
    :param mean: mean pixel vector
    :param n_pixels: total number of pixels processed so far
    :return:
    """
    # Temp variables for simplifying calculation
    z = np.expand_dims(x_i - mean, axis=1)
    A_inv = (n_pixels / (n_pixels - 1)) * inv_cov
    u = (1 / np.sqrt(n_pixels)) * z
    v_t = u.T
    a = v_t.dot(A_inv)

    # Calculate updated covariance
    new_inv_cov = A_inv - ((A_inv.dot(u) @ a) / (1 + a.dot(u)))
    return new_inv_cov


@threadpool_limits.wrap(limits=1, user_api='blas')
def calc_mahalanobis_distance(x, mean, inv_cov):
    diff = x - mean
    mahalanobis_dist = np.sqrt(np.dot(np.dot(diff, inv_cov), diff.T))
    return mahalanobis_dist
