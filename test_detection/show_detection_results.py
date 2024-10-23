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


def visualise_results(config: dict):
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
    print("Aggregating model results...")
    for filename in tqdm(glob("results/detection_results_*.pkl")):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Extract model name from results filename
        model = filename.split("results_")[1].split(".")[0]

        # Convert time to lines per second
        metrics_dict["avg_lps"] = np.mean(1 / np.array(results_dict["times"]))
        metrics_dict["std_lps"] = np.std(1 / np.array(results_dict["times"]))

        # Get detection results array/mask
        detection_results = np.nan_to_num(np.array(results_dict["md_maps"])).mean(axis=0)

        # Save prediction map mask
        threshold = config[model]["threshold"]
        plt.imsave(f"results/detection_map_{model}.png",
                   np.where(detection_results > threshold, 1, 0),
                   format="png",
                   cmap="hot")

        plt.imsave(f"results/md_map_{model}.png",
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

            # Compute Area under the ROC Curve metrics
            auc = roc_auc_score(y_true, md)
            auc_list.append(auc)

            auc_dt = calc_auc_dt(tpr, thresholds)
            auc_dt_list.append(auc_dt)

            auc_ft = calc_auc_ft(fpr, thresholds)
            auc_ft_list.append(auc_ft)

            auc_TD = (auc + auc_dt) / 2
            auc_TD_list.append(auc_TD)

            auc_BS = (auc - auc_ft + 1) / 2
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
        roc_dict[f"{model}"] = {"fpr": fpr_mean,
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
                                                columns=[model])
        # Add new column for each result dictionary after the first
        else:
            metrics_df[model] = metrics_dict

    # Transpose dataframe so that models are now rows
    metrics_df = metrics_df.T
    metrics_df.index.name = "Model"

    # Write compiled tabulated results to csv
    metrics_df.round(3).to_csv("results/formatted_results.csv")

    # Plot ROC Curve
    plt.figure(figsize=(10, 10))

    # Plotting ROC curves with downsampling
    downsampled_data = {}
    palette = sns.color_palette("Set1", n_colors=len(roc_dict)+5)  # Use Set1 color palette
    palette = [palette[i] for i in range(len(palette)) if i != 5]  # Exclude yellow
    for i, model_name in enumerate(roc_dict.keys()):
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
            colour = palette[i + 5]
            linestyle = "-"

        # Downsample the ROC curve data
        fpr, tpr = roc_dict[model_name]["fpr"], roc_dict[model_name]["tpr"]
        tpr_std = roc_dict[model_name]["tpr_std"]
        indices = np.linspace(0, len(fpr) - 1, num=1000).astype(int)
        fpr, tpr = fpr[indices], tpr[indices]

        # Plot the downsampled ROC curve
        auc = str(np.round(roc_dict[model_name]["auc"], 3))
        auc_std = str(np.round(roc_dict[model_name]["auc_std"], 3))
        sns.lineplot(x=fpr,
                     y=tpr,
                     label=f"{custom_name}",
                     color=colour,
                     linestyle=linestyle,
                     linewidth=3,
                     alpha=0.8)
        plt.fill_between(fpr, tpr-tpr_std, tpr+tpr_std, color=colour, alpha=0.3)

        # Store downsampled fpr and tpr in the dictionary
        downsampled_data[f'{model_name}_fpr'] = fpr
        downsampled_data[f'{model_name}_tpr'] = tpr

    # Convert the dictionary to a DataFrame and transpose it
    downsampled_df = pd.DataFrame(downsampled_data)

    # Save the transposed downsampled data to a CSV file
    downsampled_df.to_csv('results/roc_results.csv', index=False)

    # Plotting the diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='black', lw=3, linestyle='--', alpha=0.3)

    # Set plot limits and labels
    plt.xlim([0.0, 1.0])
    plt.xticks(fontsize=26)
    plt.ylim([0.0, 1.0])
    plt.yticks(fontsize=26)
    plt.xlabel('False Positive Rate', fontsize=26)
    plt.ylabel('True Positive Rate', fontsize=26)
    plt.legend(loc="lower right", title=None, fontsize=26, title_fontsize=26, frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Display the plot
    plt.savefig("results/roc_curve.png")
    return None


def visualise_boxplots(config: dict):
    """
        """
    # Load anomaly ground truth
    mask_dir = config["MASK_DIR"]

    # Load mask (check if reversed or not)
    if config["REVERSE"] is False:
        hsi_mask = np.load(f"../{mask_dir}")
    else:
        hsi_mask = np.flip(np.load(f"../{mask_dir}"), axis=1)

    # Loop through all model result files
    data = []
    for i, filename in tqdm(enumerate(glob("results/detection_results_*.pkl"))):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Extract model name from results filename
        model_name = filename.split("results_")[1].split(".")[0]
        # Custom model names
        if model_name == "erx":
            model_name = "ERX"
        elif model_name == "lbl_ad":
            model_name = "LBL-AD"
        elif model_name == "rx_bil":
            model_name = "RX-BIL"
        elif model_name == "rx_baseline":
            model_name = "RX Baseline"
            linestyle = ":"
        elif model_name == "rt_ck_rxd":
            model_name = "RT-CK-RXD"

        # Get detection results array/mask
        md_map = np.nan_to_num(np.array(results_dict["md_maps"])).mean(axis=0)

        # Normalize distance values to be between 0 and 1
        distance_values_normalized = (md_map - md_map.min()) / (md_map.max() - md_map.min())

        # Flatten arrays
        distance_values_flat = distance_values_normalized.flatten()
        mask_flat = hsi_mask.flatten()

        # Add to the data list
        data.append(pd.DataFrame({
            'Distance': distance_values_flat,
            'Mask': mask_flat,
            'Model': [model_name] * len(distance_values_flat)
        }))

    # Combine all DataFrames
    combined_data = pd.concat(data)

    # Set up the plot
    plt.figure(figsize=(18, 9))

    # Create boxplots with Seaborn
    sns.boxplot(x='Model',
                y='Distance',
                hue='Mask',
                data=combined_data,
                hue_order=[0, 1],  # Set order for hues
                dodge=True,  # Separate boxplots for each hue
                palette={0: 'blue', 1: 'red'},  # Automatic color palette
                boxprops=dict(alpha=.7),
                whis=(0, 100),
                showfliers=True)  # Separate boxplots for each hue

    # Customize plot aesthetics
    plt.ylabel("Normalised Mahalanobis Distance", fontsize=26)
    plt.xlabel(None, fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim(0, 1)  # Ensure the y-axis is within [0, 1]

    # Create custom legend labels and colors
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=(0, 0, 1, 0.5), label='Background'),
        plt.Rectangle((0, 0), 1, 1, color=(1, 0, 0, 0.5), label='Anomalies')
    ]
    plt.legend(handles=handles, loc='upper right', fontsize=26, frameon=True)

    # Show the plot
    plt.tight_layout()
    plt.savefig("results/boxplots.png")


def main():
    # Import test configuration file from local directory
    with open("config_detection.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Visualise numerical results and save figure
    visualise_results(config=config)
    visualise_boxplots(config=config)
    return None


if __name__ == '__main__':
    main()
