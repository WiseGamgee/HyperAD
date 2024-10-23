import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def calc_auc_dt(tpr, thresholds):
    # Normalize thresholds between 0 and 1
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    # Compute AUC between TPR and normalized thresholds using trapezoidal rule
    auc_tpr_vs_threshold = np.trapz(tpr, normalized_thresholds[::-1])
    return auc_tpr_vs_threshold


def calc_auc_ft(fpr, thresholds):
    # Normalize thresholds between 0 and 1
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    # Compute AUC between TPR and normalized thresholds using trapezoidal rule
    auc_fpr_vs_threshold = np.trapz(fpr, normalized_thresholds[::-1])
    return auc_fpr_vs_threshold


def get_box_plot(md_map, mask, save_dir: str = None):
    # Normalize the distance values to be between 0 and 1
    md_normalised = (md_map - md_map.min()) / (md_map.max() - md_map.min())

    # Flatten the 2D arrays
    distance_values_flat = md_normalised.flatten()
    mask_flat = mask.flatten()

    # Organize data for plotting
    md_normal = distance_values_flat[mask_flat == 0]
    md_anomaly = distance_values_flat[mask_flat == 1]

    # Create a figure
    plt.figure(figsize=(8, 8))

    # Create the boxplot
    sns.boxplot(data=[md_normal, md_anomaly],
                palette=[(0, 0, 1, 0.8), (1, 0, 0, 0.8)])

    # Set labels and title
    plt.xticks([0, 1], ["Normal", "Anomalies"], fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Distance Values", fontsize=26)

    # Save plot
    if save_dir:
        plt.savefig(save_dir)
