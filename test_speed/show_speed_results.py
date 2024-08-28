import yaml
import pickle
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


def visualise_results(x_axis: str):
    """
    :param: x_axis: name of the dataset parameter being used (n_bands, n_pixels or n_lines)
    :return:
    """
    results_df = None
    for filename in glob("results/results_*.pkl"):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)
            x_value = results_dict["params"][x_axis]
            results = results_dict["time"]

        # Create dataframe for first results dictionary loaded
        if results_df is None:
            results_df = pd.DataFrame.from_dict(results,
                                                orient="index",
                                                columns=[x_value])
        # Add new column for each result dictionary after the first
        else:
            results_df[x_value] = results

    # Add origin points and sort columns
    results_df[0] = 0
    results_df = results_df.reindex(sorted(results_df.columns), axis=1)

    # Convert to milliseconds
    results_df *= 1000
    results_df.to_csv("results/formatted_results.csv")

    # Specify plot params
    y_label = "Average Processing Time per Line (Milliseconds)"
    title = "Anomaly Detector Speed Comparison"
    if x_axis == "n_bands":
        x_label = "No. of Hyperspectral Bands"
    elif x_axis == "n_pixels":
        x_label = "No. of Pixels per Line"
    elif x_axis == "n_lines":
        x_label = "No. of Lines in the HSI"
    else:
        raise ValueError("X-AXIS value must be n_bands, n_pixels or n_lines.")

    # Plot results
    results_df.T.plot(figsize=(10, 10),
                      xlabel=x_label,
                      ylabel=y_label,
                      title=title)
    plt.savefig(f"results/{title}.png", format="png")
    plt.show()
    return None


def main():
    # Import test configuration file from local directory
    with open("config_speed.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Visualise numerical results and save figure
    visualise_results(x_axis=config["TEST_DIMENSION"])
    return None


if __name__ == '__main__':
    main()
