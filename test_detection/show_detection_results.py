import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, roc_auc_score, confusion_matrix


def visualise_results(config: dict):
    """
    """
    # Load anomaly ground truth
    mask_dir = config["MASK_DIR"]
    hsi_mask = np.load(f"../{mask_dir}")
    y_true = hsi_mask.flatten()

    # Loop through all model result files
    metrics_dict = {}
    metrics_df = None
    for filename in glob("results/detection_results_*.pkl"):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Extract model name from results filename
        model = filename.split("results_")[1].split(".")[0]

        # Convert time to milliseconds
        metrics_dict["time_ms"] = results_dict["time"] * 1000

        # Get detection results array/mask
        detection_results = results_dict["detection_results"]

        # Save prediction map mask
        threshold = config[model]["threshold"]
        plt.imsave(f"results/detection_map_{model}.png",
                   np.where(detection_results > threshold, 1, 0),
                   format="png",
                   cmap="hot")

        plt.imsave(f"results/md_map_{model}.png",
                   detection_results,
                   format="png",
                   cmap="hot")

        # Skip the following metrics calculations if specified (useful for only generating heatmaps)
        if config["SKIP_METRICS"]:
            continue

        # Compute Area under the ROC Curve
        md = detection_results.flatten()
        metrics_dict["roc_auc"] = roc_auc_score(y_true, md)

        # Get binary predictions and compute classification metrics
        y_pred = np.where(md > threshold, 1, 0)
        metrics_dict["f1_score"] = f1_score(y_true, y_pred)
        metrics_dict["recall"] = recall_score(y_true, y_pred)
        metrics_dict["precision"] = precision_score(y_true, y_pred)
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics_dict["true_negative"] = int(tn)
        metrics_dict["false_positive"] = int(fp)
        metrics_dict["false_negative"] = int(fn)
        metrics_dict["true_positive"] = int(tp)

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
    return None


def main():
    # Import test configuration file from local directory
    with open("config_detection.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Visualise numerical results and save figure
    visualise_results(config=config)
    return None


if __name__ == '__main__':
    main()
