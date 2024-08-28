import numpy as np
from collections import deque
from scipy.spatial import distance
from utils.simulation_core import Detector


class ERX(Detector):
    """
    A fast RX variant that updates the mean and covariance using exponentially moving statistics
    Implementation published in:

    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 momentum: float = 0.01,
                 line_offset: int = 0,
                 normalise_md: bool = False,
                 threshold: float = None,
                 classifier=None,
                 classifier_kwargs: dict = None):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param momentum: the weighting factor placed on the most recent line's mean/covariance
                1 - momentum is the weighting placed on the previous mean/covariance
        :param line_offset: no. of lines between current line and the line prediction is made for
                e.g. line_offset = 1 will predict anomalies for the second last line captured
        :param normalise_md: normalises mahalanobis distance across each line
        :param threshold: Mahalanobis distance where values above are classified as anomalies
                n.b. default is None which returns raw (or normalised) Mahalanobis distance
        :param classifier: an object or function for classifying pixels as anomalies
                from the Mahalanobis distance
        :param classifier_kwargs: dictionary containing any key inputs for the classifier
        """
        # Store inputs as attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.momentum = momentum
        self.threshold = threshold
        self.classifier = classifier
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.normalise_md = normalise_md
        self.classifier_kwargs = classifier_kwargs

        # Create buffer
        self.buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.cov = None
        self.mean = None
        self.line_count = 0
        self.I_mini = np.identity(n_bands) * 1e-5

    # Takes a line as input and outputs AD result
    def forward(self, x):
        """
        :param x: line of hyperspectral pixels with shape (n_pixels, n_bands)
        :return: anomaly predictions or raw detector outputs with shape (n_pixels, )
        """
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Get mean and covariance of current line
        line_mean = x.mean(axis=0)
        line_cov = np.cov(x, rowvar=False) + self.I_mini

        # Update running statistics
        if self.line_count == 1:
            # Initial values approximated by first line stats
            self.mean = line_mean
            self.cov = line_cov
        else:
            # Exponential moving stats
            self.mean -= (self.mean - line_mean) * self.momentum
            self.cov -= (self.cov - line_cov) * self.momentum

        # Check if buffer filled (begin anomaly detection)
        if self.line_count >= self.buffer_len:
            # Use Cholesky decomposition to get Mahalanobis distance
            delta = self.buffer[-self.line_offset - 1] - self.mean
            md = mahalanobis_cholesky(delta=delta,
                                      cov=self.cov,
                                      I_mini=self.I_mini)

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()

            # Use classifier and return anomaly predictions
            if self.classifier:
                if self.classifier_kwargs:
                    y_pred = self.classifier(md, **self.classifier_kwargs)
                else:
                    y_pred = self.classifier(md)
                return y_pred

            # # Use threshold and return anomaly predictions
            # elif self.threshold:
            #     y_pred = np.where(md > self.threshold, 1, 0)
            #     return y_pred

            # Otherwise, just return the Mahalanobis distance (raw or normalised)
            else:
                return md

        # Returns None if buffer not filled yet
        else:
            return None

    # Reset detector
    def reset(self):
        self.cov = None
        self.mean = None
        self.line_count = 0
        self.buffer = deque(maxlen=self.buffer_len)


class ERX_R(Detector):
    """
    A version of ERX that uses correlation instead of covariance when calculating Mahalanobis distance.
    Implementation published in:

    """
    def __init__(self,
                 n_bands: int,
                 n_pixels: int,
                 buffer_len: int,
                 momentum: float = 0.01,
                 line_offset: int = 0,
                 normalise_md: bool = False,
                 threshold: float = None,
                 classifier=None,
                 classifier_kwargs: dict = None):
        """
        :param n_bands: no. of hyperspectral bands/channels
        :param n_pixels: no. of pixels per line
        :param buffer_len: maximum no. of lines stored in rolling buffer
        :param momentum: the weighting factor placed on the most recent line's mean/covariance
                1 - momentum is the weighting placed on the previous mean/covariance
        :param line_offset: no. of lines between current line and the line prediction is made for
        e.g. line_offset = 1 will predict anomalies for the second last line captured
        :param normalise_md: normalises mahalanobis distance across each line
        :param threshold: Mahalanobis distance where values above are classified as anomalies
        n.b. default is None which returns raw (or normalised) Mahalanobis distance
        :param classifier: an object or function for classifying pixels as anomalies
        from the Mahalanobis distance
        :param classifier_kwargs: dictionary containing any key inputs for the classifier
        """

        # Store inputs as attributes
        self.n_bands = n_bands
        self.n_pixels = n_pixels
        self.momentum = momentum
        self.threshold = threshold
        self.classifier = classifier
        self.buffer_len = buffer_len
        self.line_offset = line_offset
        self.normalise_md = normalise_md
        self.classifier_kwargs = classifier_kwargs

        # Create buffer
        self.buffer = deque(maxlen=buffer_len)

        # Instantiate online tracking variables
        self.corr = None
        self.line_count = 0
        self.zero_mean = np.zeros((1, n_bands))
        self.I_mini = np.identity(n_bands) * 1e-5

    # Purpose of forward is to take line input and output AD result
    def forward(self, x):
        """
        :param x: line of hyperspectral pixels with shape (n_pixels, n_bands)
        :return: anomaly predictions or raw detector outputs with shape (n_pixels, )
        """
        # Append line to buffer
        self.line_count += 1
        self.buffer.append(x)

        # Get correlation of current line
        line_corr = (x.T @ x) / self.n_pixels

        # Update running statistics
        if self.line_count == 1:
            # Initial stats approximated by first line
            self.corr = line_corr
        else:
            # Exponential moving stats
            self.corr -= (self.corr - line_corr) * self.momentum

        # Get inverse of correlation matrix for md calcs
        inv_corr = np.linalg.inv(self.corr)

        # Check if buffer filled
        if self.line_count >= self.buffer_len:
            # Get mahalanobis distance
            md = distance.cdist(self.buffer[-self.line_offset - 1],
                                self.zero_mean,
                                metric='mahalanobis',
                                VI=inv_corr).flatten()

            # If specified, normalise current line md using the line mean and standard dev
            if self.normalise_md:
                md = (md - md.mean()) / md.std()

            # Use classifier and return anomaly predictions
            if self.classifier:
                if self.classifier_kwargs:
                    y_pred = self.classifier(md, **self.classifier_kwargs)
                else:
                    y_pred = self.classifier(md)
                return y_pred

            # # Use threshold and return anomaly predictions
            # elif self.threshold:
            #     y_pred = np.where(md > self.threshold, 1, 0)
            #     return y_pred

            # Otherwise, just return the Mahalanobis distance (raw or normalised)
            else:
                return md

            # Returns None if buffer not filled yet
        else:
            return None

    # Reset detector
    def reset(self):
        self.corr = None
        self.line_count = 0
        self.buffer = deque(maxlen=self.buffer_len)


def mahalanobis_cholesky(delta, cov, I_mini):
    # Decompose covariance into lower triangle
    L = np.linalg.cholesky(cov + I_mini)

    # Use forward substitution and norm to solve for Mahalanobis dist
    Z = np.linalg.solve(L, delta.T)
    return np.linalg.norm(Z, axis=0)
