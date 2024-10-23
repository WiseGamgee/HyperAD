import numpy as np
from collections import deque
from scipy.spatial import distance
from utils.simulation_core import Detector


class RXBaseline(Detector):
    """
    A very basic dynamic, real-time model implementation for detecting anomalies.
    Used as a baseline approach for comparing performance to other methods.
    Implementation published in:
    Garske, Samuel and Evans, Bradley and Artlett, Christopher and Wong, KC, 2024.
    ERX - a Fast Real-Time Anomaly Detection Algorithm for Hyperspectral Line-Scanning
    arXiv preprint arXiv:2408.14947
    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 pixel_dropout: float = None,
                 bands_dropout: float = None,
                 line_offset: int = 0,
                 normalise_md: bool = False):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param pixel_dropout: fraction of pixels to be dropped out
        :param bands_dropout: fraction of bands to be dropped out
        :param line_offset: no. of lines between current line and the line prediction is made for
                e.g. line_offset = 1 will predict anomalies for the second last line captured
        :param normalise_md: normalises mahalanobis distance across each line
        """

        # Store inputs as attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.pixel_dropout = pixel_dropout
        self.bands_dropout = bands_dropout
        self.normalise_md = normalise_md

        # Create buffer and line count tracker
        self.line_count = 0
        self.buffer = deque(maxlen=buffer_len)

    def forward(self, x):
        # Append new line to buffer
        self.buffer.append(x)
        self.line_count += 1

        # Check if buffer filled
        if self.line_count >= self.buffer_len:
            # Convert line buffer into pixel array
            pixel_array = np.array(self.buffer).reshape(-1, self.n_bands)

            # Drop pixels from array if specified
            if self.pixel_dropout:
                indices = np.random.choice(np.arange(len(pixel_array)),
                                           size=int((1 - self.pixel_dropout) * len(pixel_array)),
                                           replace=False)
                pixel_array = pixel_array[indices]

            # Drop bands from array if specified
            if self.bands_dropout:
                bands = np.random.choice(np.arange(self.n_bands),
                                         size=int((1 - self.bands_dropout) * self.n_bands),
                                         replace=False)
                pixel_array = pixel_array[:, bands]

            # Calculate mean and covariance across pixel bands
            mean = pixel_array.mean(axis=0)
            cov = np.cov(pixel_array, rowvar=False)
            inv_cov = np.linalg.inv(cov)

            # Get mahalanobis distance
            md = distance.cdist(self.buffer[-self.line_offset - 1],
                                mean.reshape(1, -1),
                                metric='mahalanobis',
                                VI=inv_cov).flatten()

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()
            return md

            # Returns None if buffer not filled yet
        else:
            return None

    def reset(self):
        self.line_count = 0
        self.buffer = deque(maxlen=self.buffer_len)
