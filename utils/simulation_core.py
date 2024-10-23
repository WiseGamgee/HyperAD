import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from timeit import default_timer as timer


class Detector(ABC):
    """
    A base class for all line anomaly detectors
    """
    # Takes a line as input, updates model, and returns detection results
    @abstractmethod
    def forward(self, x):
        pass

    # Resets detector to initial state (i.e. zero params, emtpy buffer, etc.)
    @abstractmethod
    def reset(self):
        pass


class Sampler:
    """
    A very generic object for sampling from a pre-collected dataset
    Generally used to sample from a hyperspectral image a line at a time
    """
    def __init__(self, array, axis=0, reverse=False):
        """
        :param array: numpy array containing dataset
        :param axis: the axis for which the samples are from
        :param reverse: flips the order which samples are drawn (i.e. flip image)
        """
        # Store attributes
        self.axis = axis
        self.array = array
        self.len = array.shape[axis]
        self.reverse = reverse

        # Index for tracking samples
        self.idx = 0

    def get_sample(self):
        """
        :return: next sample from dataset
        """
        # Get next sample
        if self.reverse is False:
            sample = self.array.take(indices=self.idx, axis=self.axis)
        else:
            sample = self.array.take(indices=-self.idx, axis=self.axis)

        # Increment index to next position, reset if end of array
        self.idx += 1
        if self.idx >= self.len:
            self.idx = 0
        return sample

    def reset(self):
        """
        Return to start of dataset
        """
        self.idx = 0


class DetectionSimulator:
    """
    A simulator that feeds hyperspectral lines from the sampler to the detector
    """
    def __init__(self, sampler, detector):
        """
        sampler: object that simulates pushbroom camera capture
        detecter: algorithm that takes samples and performs detection
        """
        self.sampler = sampler
        self.detector = detector

        self.lines = sampler.len
        self.sampler_size = [sampler.array.shape[0], sampler.array.shape[1]]

        # Get line offset from detector if available, otherwise set to 0
        self.line_offset = getattr(self.detector, 'line_offset', 0)

    def run(self, disable_progress_bar=False, save_results=True):
        """
        :param disable_progress_bar: toggle tqdm progress bar on/off
        :param save_results: toggle storing detection results for testing
        :return: anomaly detection results and average time to process each line
        """
        # Create empty array to store all anomaly detection results
        if save_results:
            detection_results = np.zeros(shape=self.sampler_size)
        else:
            detection_results = None

        # Reset objects
        self.sampler.reset()
        self.detector.reset()

        # Loop through all samples sequentially (causal system)
        avg_processing_time = 0
        for i in tqdm(range(self.lines), disable=disable_progress_bar):
            # Get current sample
            hs_line = self.sampler.get_sample()

            # Start timer
            start_time = timer()

            # Get detected anomalies and update model
            y_pred = self.detector.forward(hs_line)

            # Accumulate processing time for current line
            end_time = timer()
            avg_processing_time += end_time - start_time

            # Check if results are present
            # If no results predictions left as zero
            if y_pred is not None and save_results is True:
                # Store detections in array
                detection_results[:, i - self.line_offset] = y_pred

        # Calculate average time for detector to process a line
        avg_processing_time /= self.lines

        # Return results
        return detection_results, avg_processing_time


def get_predictions_from_md(md: np.ndarray,
                            normalise_md: bool = False,
                            threshold: float = None,
                            classifier=None,
                            classifier_kwargs: dict = None):
    """
    A common detector function that performs detector agnostic processing of the Mahalanobis distance
    Used to get the anomaly predictions from the MD, or if no threshold/classifier provided then returns MD
    :param md: 1D array of Mahalanobis distance values for a line of pixels
    :param normalise_md: normalises mahalanobis distance across each line
    :param threshold: Mahalanobis distance where values above are classified as anomalies
            n.b. default is None which returns raw (or normalised) Mahalanobis distance
    :param classifier: an object or function for classifying pixels as anomalies
            from the Mahalanobis distance
    :param classifier_kwargs: dictionary containing any key inputs for the classifier
    :return:
    """
    # If specified, normalise current line md using the line mean and standard dev
    if normalise_md:
        md = (md - md.mean()) / md.std()

    # Use classifier and return anomaly predictions
    if classifier:
        if classifier_kwargs:
            y_pred = classifier(md, **classifier_kwargs)
        else:
            y_pred = classifier(md)
        return y_pred

    # Use threshold and return anomaly predictions
    elif threshold:
        y_pred = np.where(md > threshold, 1, 0)
        return y_pred

    # Otherwise, just return the Mahalanobis distance (raw or normalised)
    else:
        return md
