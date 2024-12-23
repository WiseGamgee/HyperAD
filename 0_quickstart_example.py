import numpy as np
import matplotlib.pyplot as plt
from detectors.erx import ERX


def quickstart_demo():
    """Demonstrates the ERX line-scan anomaly detection on synthetic data.

    Creates a synthetic 3D hyperspectral image with an anomalous square region,
    then processes it line-by-line as would occur in a real line-scan camera.
    The output shows the anomaly scores where brighter regions indicate anomalies.

    Note: In real-time applications, this would typically use a while loop to
    continuously process incoming camera lines. See 1_real_time_example.py for an
    example of real-time implementation.
    """
    # Create synthetic hyperspectral image (500 pixels × 300 lines × 90 spectral bands)
    dummy_camera = np.random.rand(500, 300, 90)

    # Insert an anomalous square region (higher intensity values)
    dummy_camera[275:300, 150:175, :] += 1

    # Initialize ERX model with parameters:
    # - n_bands: Number of spectral bands in the data
    # - n_pixels: Width of each line scan in pixels
    # - buffer_len: Number of previous lines for background estimation
    # - n_projdims: Number of projection dimensions for dimensionality reduction
    # - momentum: Learning rate for updating background statistics
    model = ERX(n_bands=90,
                n_pixels=500,
                buffer_len=99,
                n_projdims=5,
                momentum=0.1)

    # Process image line by line
    img = []
    for i in range(dummy_camera.shape[1]):
        # Get next line from camera (500 pixels × 90 bands)
        x = dummy_camera[:, i]

        # Process line using ERX model
        # Returns anomaly scores (Mahalanobis distances)
        y = model.forward(x)

        # Store anomaly scores (model returns None during initial buffer filling)
        img.append(y if y is not None else np.zeros(500))

    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.imshow(np.array(img).T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.xlabel('Line Number')
    plt.ylabel('Pixel Number')
    plt.title('ERX Anomaly Detection Results')
    plt.show()


if __name__ == "__main__":
    quickstart_demo()
