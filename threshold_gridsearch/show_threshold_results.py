import yaml
import pickle
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, precision_score, \
    f1_score, roc_auc_score, confusion_matrix


def visualise_results(config: dict):
    """
    :param: x_axis: name of the dataset parameter being used (n_bands, n_pixels or n_lines)
    :return:
    """
    # Load anomaly ground truth
    mask_dir = config["MASK_DIR"]
    hsi_mask = np.load(f"../{mask_dir}")
    y_true = hsi_mask.flatten()

    # Loop through all model result files
    for filename in tqdm(glob("results/threshold_results_*.pkl")):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Get detection results array
        detection_results = results_dict[f"detection_results"]

        # Flatten prediction mask for metric calculations
        md = detection_results.flatten()

        # Extract model name from results filename
        model_name = filename.split("results_")[1].split(".")[0]

        # Create dataframe for all threshold results
        metrics_df = None
        for threshold in config["THRESHOLD_LIST"]:
            # Empty dict for storing current metrics
            metrics_dict = {}

            # Compute Area under the ROC Curve
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
                                                    columns=[threshold])
            # Add new column for each result dictionary after the first
            else:
                metrics_df[threshold] = metrics_dict

        # Transpose dataframe so that models are now rows
        results_df = metrics_df.T
        results_df.index.name = "Threshold"

        # Write compiled tabulated results to csv
        results_df.round(3).to_csv(f"results/formatted_results_{model_name}.csv")
    return None


def main():
    # Import test configuration file from local directory
    with open("config_threshold.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Visualise numerical results and save figure
    visualise_results(config=config)
    return None


if __name__ == '__main__':
    main()
