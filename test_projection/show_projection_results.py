import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from utils.metrics import calc_auc_dt, calc_auc_ft
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, roc_curve, roc_auc_score, confusion_matrix


def visualise_detection_results(config: dict):
    """
    """
    # Load anomaly ground truth
    mask_dir = config["MASK_DIR"]

    # Load mask (check if reversed or not)
    if config["REVERSE"] is False:
        hsi_mask = np.load(f"../{mask_dir}")
    else:
        hsi_mask = np.flip(np.load(f"../{mask_dir}"), axis=1)
    y_true = hsi_mask.flatten()

    # Loop through all model result files
    roc_dict = {}
    metrics_dict = {}
    metrics_df = None
    threshold = config["THRESHOLD"]
    for filename in tqdm(glob("results/projection_results_*.pkl")):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Extract model name from results filename
        n_projdims = results_dict["n_projdims"]
        if n_projdims is not None:
            n_projdims = int(n_projdims)
            model_name = n_projdims
        else:
            n_projdims = 999
            model_name = "No SRP"

        # Convert time to lines per second
        metrics_dict["avg_lps"] = np.mean(1 / np.array(results_dict["times"]))
        metrics_dict["std_lps"] = np.std(1 / np.array(results_dict["times"]))

        # Get detection results array/mask
        detection_results = np.nan_to_num(np.array(results_dict["md_maps"])).mean(axis=0)

        # Save prediction map mask
        plt.imsave(f"results/detection_map_{model_name}.png",
                   np.where(detection_results > threshold, 1, 0),
                   format="png",
                   cmap="hot")

        plt.imsave(f"results/md_map_{model_name}.png",
                   detection_results,
                   format="png",
                   cmap="turbo")

        # Skip the following metrics calculations if specified (useful for only generating heatmaps)
        if config["SKIP_METRICS"]:
            continue

        # Loop through all detection maps and calculate metrics
        tpr_list, fpr_list, auc_list = [], [], []
        auc_dt_list, auc_ft_list, auc_TD_list, auc_BS_list = [], [], [], []
        f1_list, precision_list, recall_list = [], [], []
        tp_list, fp_list, tn_list, fn_list = [], [], [], []
        accuracy_list = []
        for md_map in results_dict["md_maps"]:
            md = np.nan_to_num(md_map).flatten()

            # Compute ROC curve
            fpr, tpr, thresholds = roc_curve(y_true, md)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            # Compute Area under the ROC Curve
            auc = roc_auc_score(y_true, md)
            auc_list.append(auc)

            auc_dt = calc_auc_dt(tpr, thresholds)
            auc_dt_list.append(auc_dt)

            auc_ft = calc_auc_ft(fpr, thresholds)
            auc_ft_list.append(auc_ft)

            auc_TD = auc + auc_dt
            auc_TD_list.append(auc_TD)

            auc_BS = auc - auc_ft
            auc_BS_list.append(auc_BS)

            # Get binary predictions and compute classification metrics
            y_pred = np.where(md > threshold, 1, 0)
            f1_list.append(f1_score(y_true, y_pred))
            precision_list.append(precision_score(y_true, y_pred))
            recall_list.append(recall_score(y_true, y_pred))
            accuracy_list.append(accuracy_score(y_true, y_pred))

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            tn_list.append(int(tn))
            fp_list.append(int(fp))
            fn_list.append(int(fn))
            tp_list.append(int(tp))

        # Common FPR points for interpolation (e.g., 100 evenly spaced points)
        fpr_mean = np.linspace(0, 1, 1000)

        # Interpolate TPR values for each iteration at common FPR points
        tprs_interpolated = []
        for fpr, tpr in zip(fpr_list, tpr_list):
            tpr_interpolated = np.interp(fpr_mean, fpr, tpr)
            tprs_interpolated.append(tpr_interpolated)

        # Convert list to array for easier computation
        tprs_interpolated = np.array(tprs_interpolated)

        # Calculate the mean and standard deviation of TPR across iterations
        tpr_mean = np.mean(tprs_interpolated, axis=0)
        tpr_std = np.std(tprs_interpolated, axis=0)

        # Get aggregate roc results
        metrics_dict["mean_auc"] = np.mean(auc_list)
        metrics_dict["std_auc"] = np.std(auc_list)
        roc_dict[n_projdims] = {"fpr": fpr_mean,
                                "tpr": tpr_mean,
                                "tpr_std": tpr_std,
                                "auc": np.mean(auc_list),
                                "auc_std": np.std(auc_list)}

        # Additional AUC metrics
        metrics_dict["mean_auc_dt"] = np.mean(auc_dt_list)
        metrics_dict["std_auc_dt"] = np.std(auc_dt_list)
        metrics_dict["mean_auc_ft"] = np.mean(auc_ft_list)
        metrics_dict["std_auc_ft"] = np.std(auc_ft_list)
        metrics_dict["mean_auc_td"] = np.mean(auc_TD_list)
        metrics_dict["std_auc_td"] = np.std(auc_TD_list)
        metrics_dict["mean_auc_bs"] = np.mean(auc_BS_list)
        metrics_dict["std_auc_bs"] = np.std(auc_BS_list)

        # Get binary predictions and compute classification metrics
        metrics_dict["f1_score"] = np.mean(f1_list)
        metrics_dict["recall"] = np.mean(recall_list)
        metrics_dict["precision"] = np.mean(precision_list)
        metrics_dict["accuracy"] = np.mean(accuracy_list)

        metrics_dict["true_negative"] = int(np.mean(tn_list))
        metrics_dict["false_positive"] = int(np.mean(fp_list))
        metrics_dict["false_negative"] = int(np.mean(fn_list))
        metrics_dict["true_positive"] = int(np.mean(tp_list))

        # Create dataframe for first results dictionary loaded
        if metrics_df is None:
            metrics_df = pd.DataFrame.from_dict(metrics_dict,
                                                orient="index",
                                                columns=[n_projdims])
        # Add new column for each result dictionary after the first
        else:
            metrics_df[n_projdims] = metrics_dict

    # Transpose dataframe so that models are now rows
    metrics_df = metrics_df.T
    metrics_df.index.name = "n_projdims"

    # Write compiled tabulated results to csv
    metrics_df.round(3).to_csv("results/formatted_results.csv")

    # ------------------------------------------------------------------------------------------------------------------
    # Plot ROC Curve
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 10))

    # Plotting ROC curves with downsampling
    downsampled_data = {}
    palette = sns.color_palette("Set1", n_colors=len(roc_dict)*2)  # Use Set1 color palette
    palette = [palette[i] for i in range(len(palette)) if i != 5]  # Exclude yellow
    linestyles = ['-', '--', '-.', ':']
    for i, n_projdims in enumerate(sorted(roc_dict.keys())): #enumerate([1, 3, 5, 10, 20, 30, 50, 999]): #
        colour = palette[i]
        linestyle = linestyles[i % len(linestyles)]

        # Downsample the ROC curve data
        fpr, tpr = roc_dict[n_projdims]["fpr"], roc_dict[n_projdims]["tpr"]
        tpr_std = roc_dict[n_projdims]["tpr_std"]
        indices = np.linspace(0, len(fpr) - 1, num=1000).astype(int)
        fpr, tpr = fpr[indices], tpr[indices]

        # Plot the downsampled ROC curve
        #auc = str(np.round(roc_dict[n_projdims]["auc"], 3))
        #auc_std = str(np.round(roc_dict[n_projdims]["auc_std"], 3))
        if n_projdims == 999:
            n_projdims = "No SRP"
        sns.lineplot(x=fpr,
                     y=tpr,
                     label=f"{n_projdims}",
                     color=colour,
                     linestyle=linestyle,
                     linewidth=3,
                     alpha=0.8)

        # Plot error boundaries
        if config["SHOW_STD_ON_ROC"]:
            plt.fill_between(fpr, tpr - tpr_std, tpr + tpr_std, color=colour, alpha=0.3)

        # Store downsampled fpr and tpr in the dictionary
        downsampled_data[f'{n_projdims}_fpr'] = fpr
        downsampled_data[f'{n_projdims}_tpr'] = tpr

    # Convert the dictionary to a DataFrame and transpose it
    downsampled_df = pd.DataFrame(downsampled_data)

    # Save the transposed downsampled data to a CSV file
    downsampled_df.to_csv('results/roc_results.csv', index=False)

    # Plotting the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', alpha=0.3)

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=26)
    plt.ylim([0.0, 1.0])
    plt.yticks(fontsize=26)
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.legend(loc="lower right", title=r"SRP Dims ($d$)", fontsize=26, title_fontsize=26, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.savefig("results/roc_curve.png")

    # ------------------------------------------------------------------------------------------------------------------
    # Plot speed curve
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a DataFrame with filtered data
    speed_df = metrics_df[["avg_lps", "std_lps"]].sort_index()

    # Plotting speed curves
    sns.lineplot(x=speed_df.index, y=speed_df["avg_lps"], alpha=0.8)
    plt.fill_between(speed_df.index,
                     speed_df['avg_lps'] - speed_df['std_lps'], speed_df['avg_lps'] + speed_df['std_lps'], alpha=0.3)

    # Set plot limits and labels
    ax.set_xlabel("Number of Projected Dimensions", fontsize=22)
    ax.tick_params(axis='x', labelsize=20)

    # Set linear ticks for x-axis
    linear_ticks = np.arange(0, max(speed_df.index) + 5, step=10)
    ax.set_xticks(linear_ticks)
    ax.set_xticklabels([f"{int(x)}" for x in linear_ticks])

    # Set y-axis to log scale
    #ax.set_yscale("log")
    ax.set_ylabel('Speed (Lines per Second)', fontsize=22)
    ax.tick_params(axis='y', labelsize=20)

    # Set limits for y-axis
    ax.set_ylim(0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.savefig("results/dataset_speed_comparison.png")
    return None


def visualise_speed_results(config: dict):
    """
    """
    for filename in glob("results/speed_results_*.pkl"):
        with open(filename, 'rb') as f:
            speed_dict = pickle.load(f)
            lps_dict = {int(key): np.mean(1 / np.array(value)) for key, value in speed_dict.items()}
            std_dict = {int(key): np.std(1 / np.array(value)) for key, value in speed_dict.items()}

            # Convert time to lines per second
            lps_df = pd.DataFrame(data={"avg_lps": lps_dict, "std_lps": std_dict})

    # ------------------------------------------------------------------------------------------------------------------
    # Plot speed curve
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting speed curves
    sns.lineplot(x=lps_df.index,
                 y=lps_df["avg_lps"],
                 alpha=0.8)
    plt.fill_between(lps_df.index,
                     lps_df["avg_lps"] - lps_df["std_lps"],
                     lps_df["avg_lps"] + lps_df["std_lps"],
                     alpha=0.3)

    # Set plot limits and labels
    ax.set_xlabel("Number of Projected Dimensions", fontsize=16)
    ax.tick_params(axis='x', labelsize=14)

    # Set linear ticks for x-axis
    linear_ticks = np.arange(0, max(lps_df.index) + 20, step=20)
    ax.set_xticks(linear_ticks)
    ax.set_xticklabels([f"{int(x)}" for x in linear_ticks])  # Set linear x-ticks

    # Set y-axis to log scale
    # ax.set_yscale("log")
    ax.set_ylabel('Speed (Lines per Second)', fontsize=16)
    ax.tick_params(axis='y', labelsize=14)

    # Set limits for y-axis
    ax.set_ylim(0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.savefig("results/general_speed_comparison.png")


def main():
    # Import test configuration file from local directory
    with open("config_projection.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Visualise numerical results and save figure
    visualise_detection_results(config=config)
    if config["RUN_SPEED_TEST"]:
        visualise_speed_results(config=config)
    return None


if __name__ == '__main__':
    main()
