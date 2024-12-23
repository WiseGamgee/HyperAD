import numpy as np
import matplotlib.pyplot as plt
from detectors.erx import ERX


def real_time_demo(data_path=None):
    """Visual demonstration of ERX line-scan anomaly detection in a real-time setting.

    Args:
        data_path: Optional path to .npy file containing hyperspectral data
                  Expected shape: (pixels, lines, bands)
                  If None, uses synthetic data
    """
    if data_path is None:
        n_pixels, n_lines, n_bands = 500, 1000, 90
        # Create synthetic data if no path provided
        dummy_camera = np.random.rand(n_pixels, n_lines, n_bands)
        # Add anomalous region
        dummy_camera[275:300, 150:175, :] += 5
    else:
        # Load data from file
        dummy_camera = np.load(data_path)
        n_pixels, n_lines, n_bands = dummy_camera.shape

    # Initialize ERX model
    model = ERX(n_bands=n_bands,
                n_pixels=n_pixels,
                buffer_len=99,
                n_projdims=5,
                momentum=0.1)

    # Set up real-time plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    img_plot = ax.imshow(np.zeros((n_pixels, 1)).T, aspect='auto', cmap='viridis',
                         vmin=0, vmax=10)  # Set fixed scale range
    plt.colorbar(img_plot, label='Anomaly Score')
    ax.set_xlabel('Line Number')
    ax.set_ylabel('Pixel Number')
    ax.set_title('ERX Real-Time Anomaly Detection')

    # Process lines in real-time
    i = 0
    img = []
    running = True

    # Keep track of min/max for color scaling
    global_min = float('inf')
    global_max = float('-inf')

    try:
        while running:
            # Get next line from camera
            x = dummy_camera[:, i]

            # Process line and get anomaly scores
            y = model.forward(x)

            # Ensure y is a numpy array and update global min/max
            if y is not None:
                global_min = min(global_min, y.min())
                global_max = max(global_max, y.max())
                img.append(y)
            else:
                # If empty then return zeros
                img.append(np.zeros(n_pixels))

            # Update visualization every 50 lines
            if i % 50 == 0:
                img_array = np.array(img).T
                img_plot.set_array(img_array)
                img_plot.set_extent([0, len(img), 0, n_pixels])

                # Update color scale using global min/max
                img_plot.set_clim(global_min, global_max)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            if i + 1 >= dummy_camera.shape[1]:
                running = False
            else:
                i += 1

    except KeyboardInterrupt:
        print("Processing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")

    # Final visualization with full range
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    real_time_demo(data_path=None)
