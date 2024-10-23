import os
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from utils.simulation_core import Sampler, DetectionSimulator
from detectors import erx_ablation


def get_accuracy_results(config: dict):
    """
    :param config: configuration dictionary
    :return:
    """
    # Load hsi dataset from file
    hsi_dir = config["HSI_DIR"]
    hsi_array = np.load(f"../{hsi_dir}")
    n_pixels, n_lines, n_bands = hsi_array.shape

    # Create hsi line sampler from dummy dataset
    line_sampler = Sampler(array=hsi_array, axis=1, reverse=config["REVERSE"])
    line_sampler.reset()

    # Iterate over each detector for evaluation
    results_dict = {}
    for momentum in tqdm(config["MOMENTUM_LIST"], disable=True):
        # Indicate current detector being evaluated
        print(f"ERX being evaluated with a momentum of: {momentum}")

        # Configure ERX detector
        erx_detector = erx_ablation.ERXwAblation(n_bands=n_bands,
                                                 n_pixels=n_pixels,
                                                 buffer_len=config["BUFFER_LEN"],
                                                 n_projdims=config["PROJECTED_DIMENSIONS"],
                                                 momentum=momentum,
                                                 line_offset=config["LINE_OFFSET"],
                                                 normalise_md=config["NORMALISE_MD"])

        # Repeat experiment multiple times if specified
        processing_time_list = []
        detection_results_list = []
        n_repeats = config["N_REPEATS"]
        for i in range(n_repeats):
            # Create simulator and run simulation
            sim = DetectionSimulator(sampler=line_sampler,
                                     detector=erx_detector)
            md_map, seconds_per_line = sim.run(disable_progress_bar=False)

            # Accumulate processing speeds across each iteration
            processing_time_list.append(seconds_per_line)
            detection_results_list.append(md_map)

        # Save processing times and md results for current detector
        results_dict["momentum"] = momentum
        results_dict["times"] = processing_time_list
        results_dict["md_maps"] = detection_results_list

        # Save detection results
        # Create local results directory if folder is missing
        if os.path.exists("results") is False:
            os.makedirs("results")

        # Save results after every model to minimise loss if error occurs
        filename = f"results/momentum_results_{momentum}"
        with open(fr"{filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)


def main():
    # Import test configuration file from local directory
    with open("config_momentum.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Run experiments and save numerical results
    get_accuracy_results(config=config)
    return None


if __name__ == '__main__':
    main()
