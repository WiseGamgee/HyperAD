import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from matplotlib import ticker
import matplotlib.pyplot as plt


def visualise_results(x_axis: str, drop_first_run: bool = False):
    """
    :param: x_axis: name of the dataset parameter being used (n_bands, n_pixels or n_lines)
    :return:
    """
    lps_df = None
    std_df = None
    for filename in glob("results/results_*.pkl"):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)
            x_value = results_dict["params"][x_axis]
            speed_results = {key: np.mean(1 / np.array(value)) for key, value in results_dict["time"].items()}
            std_results = {key: np.std(1 / np.array(value)) for key, value in results_dict["time"].items()}

        # Create dataframe for first results dictionary loaded
        if lps_df is None:
            lps_df = pd.DataFrame.from_dict(speed_results,
                                              orient="index",
                                              columns=[x_value])
            std_df = pd.DataFrame.from_dict(std_results,
                                            orient="index",
                                            columns=[x_value])
        # Add new column for each result dictionary after the first
        else:
            lps_df[x_value] = speed_results
            std_df[x_value] = std_results

    # Add origin points and sort columns
    lps_df = lps_df.reindex(sorted(lps_df.columns), axis=1)
    std_df = std_df.reindex(sorted(std_df.columns), axis=1)
    if drop_first_run:
        lps_df = lps_df.iloc[:, 1:]
        std_df = std_df.iloc[:, 1:]

    # Save results
    lps_df.to_csv("results/formatted_results.csv")

    # Specify plot params
    if x_axis == "n_bands":
        x_label = "Number of Image Bands"
    elif x_axis == "n_pixels":
        x_label = "Number of Pixels per Line"
    elif x_axis == "n_lines":
        x_label = "Number of Lines in the Image"
    else:
        raise ValueError("X-AXIS value must be n_bands, n_pixels or n_lines.")

    # Calculate average lines processed per second
    lps_df = lps_df.T.fillna(0)
    std_df = std_df.T.fillna(0)

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each model
    palette = sns.color_palette("Set1", n_colors=lps_df.shape[1] + 5)  # Use Set1 color palette
    palette = [palette[i] for i in range(len(palette)) if i != 5]  # exclude yellow
    for i, model_name in enumerate(lps_df.columns):
        # Custom model names
        if model_name == "erx":
            custom_name = "ERX"
            colour = palette[0]
            linestyle = "-"
        elif model_name == "lbl_ad":
            custom_name = "LBL-AD"
            colour = palette[1]
            linestyle = "--"
        elif model_name == "rx_bil":
            custom_name = "RX-BIL"
            colour = palette[2]
            linestyle = "-."
        elif model_name == "rx_baseline":
            custom_name = "RX Baseline"
            colour = palette[3]
            linestyle = ":"
        elif model_name == "rt_ck_rxd":
            custom_name = "RT-CK-RXD"
            colour = palette[4]
            linestyle = "-"
        else:
            custom_name = model_name
            colour = palette[i+5]
            linestyle = "-"
        sns.lineplot(x=lps_df.index,
                     y=lps_df[model_name],
                     label=custom_name,
                     color=colour,
                     linestyle=linestyle,
                     linewidth=3,
                     alpha=0.8)
        plt.fill_between(lps_df.index,
                         lps_df[model_name]-std_df[model_name],
                         lps_df[model_name]+std_df[model_name],
                         color=colour,
                         alpha=0.3)

    # Set plot limits and labels
    ax.set_xlabel(x_label, fontsize=18)
    ax.tick_params(axis='x', labelsize=18)
    ax.legend(title=None, fontsize=14, title_fontsize=16, loc='upper right')

    # Set linear ticks for x-axis
    linear_ticks = np.arange(0, lps_df.index.max() + 20, step=20)
    #linear_ticks = np.arange(0, 1600, step=300)
    ax.set_xticks(linear_ticks)
    ax.set_xticklabels([f"{int(x)}" for x in linear_ticks])

    # Set y-axis to log scale
    ax.set_yscale("log")
    ax.set_ylabel('Speed (Lines per Second)', fontsize=18)
    ax.tick_params(axis='y', labelsize=18)

    # Set limits for y-axis
    ax.set_ylim(1, 1e4)

    # Apply ScalarFormatter to ensure normal form on both axes
    formatter = ticker.ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.gca().yaxis.set_major_formatter(formatter)
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    plt.minorticks_on()
    plt.tight_layout()

    # Display the plot
    plt.savefig("results/speed_comparison")
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
