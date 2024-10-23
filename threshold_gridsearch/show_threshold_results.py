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

    # Load mask (check if reversed or not)
    if config["REVERSE"] is False:
        hsi_mask = np.load(f"../{mask_dir}")
    else:
        hsi_mask = np.flip(np.load(f"../{mask_dir}"), axis=1)
    y_true = hsi_mask.flatten()

    # Loop through all model result files
    for filename in glob("results/threshold_results_*.pkl"):
        with open(filename, 'rb') as f:
            results_dict = pickle.load(f)

        # Get detection results array
        detection_results = results_dict[f"md_maps"]

        # Flatten prediction mask for metric calculations
        md = np.nan_to_num(detection_results.flatten())

        # Extract model name from results filename
        model_name = filename.split("results_")[1].split(".")[0]

        # Create dataframe for all threshold results
        metrics_df = None
        print(f"Calculating {model_name} threshold results...")
        for threshold in tqdm(config["THRESHOLD_LIST"]):
            # Empty dict for storing current metrics
            metrics_dict = {}

            # Skip the following metrics calculations if specified (useful for only generating heatmaps)
            if config["SKIP_METRICS"]:
                continue

            # Loop through all detection maps and calculate metrics
            tpr_list, fpr_list, auc_list = [], [], []
            f1_list, precision_list, recall_list = [], [], []
            tp_list, fp_list, tn_list, fn_list = [], [], [], []
            accuracy_list = []
            for md_map in results_dict["md_maps"]:
                md = np.nan_to_num(md_map).flatten()

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
