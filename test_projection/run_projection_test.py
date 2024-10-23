import os
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from utils.simulation_core import Sampler, DetectionSimulator
from detectors import erx_ablation


#
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
    for n_projdims in tqdm(config["DIMENSION_LIST"], disable=True):
        # Indicate current detector being evaluated
        print(f"ERX being evaluated when projecting the bands to {n_projdims} dimensions.")

        # Configure ERX detector
        erx_detector = erx_ablation.ERXwAblation(n_bands=n_bands,
                                                 n_pixels=n_pixels,
                                                 buffer_len=config["BUFFER_LEN"],
                                                 n_projdims=n_projdims,
                                                 momentum=config["MOMENTUM"],
                                                 line_offset=config["LINE_OFFSET"],
                                                 normalise_md=config["NORMALISE_MD"])

        # Repeat experiment multiple times if specified
        results_dict = {}
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

        # Save processing times and md results for current number of dimensions
        results_dict["n_projdims"] = n_projdims
        results_dict["times"] = processing_time_list
        results_dict["md_maps"] = detection_results_list

        # Save detection results
        # Create local results directory if folder is missing
        if os.path.exists("results") is False:
            os.makedirs("results")

        # Save results after every model to minimise loss if error occurs
        filename = f"results/projection_results_{n_projdims}"
        with open(fr"{filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)
        del results_dict


def get_speed_results(config: dict):
    """
    :param: config: dictionary with configuration file inputs
    :return:
    """
    #
    n_pixels = config["N_PIXELS"]
    n_lines = config["N_LINES"]
    n_bands = config["N_BANDS"]

    # Create hsi line sampler from dummy dataset (only use current number of pixels, lines and bands)
    line_sampler = Sampler(array=(np.random.rand(n_pixels, n_lines, n_bands) * 10000).astype(int), axis=1)

    # Perform grid search across all projected dimensions
    idx = 0
    speed_dict = {}
    for n_projdims in tqdm(config["SPEED_DIMENSION_LIST"], disable=True):
        # Indicate current detector being evaluated
        print(f"ERX being evaluated when projecting the bands to {n_projdims} dimensions.")

        # Configure ERX detector
        erx_detector = erx_ablation.ERXwAblation(n_bands=n_bands,
                                                 n_pixels=n_pixels,
                                                 buffer_len=config["BUFFER_LEN"],
                                                 n_projdims=n_projdims,
                                                 momentum=config["MOMENTUM"],
                                                 line_offset=config["LINE_OFFSET"],
                                                 normalise_md=config["NORMALISE_MD"])

        # Repeat experiment multiple times if specified
        processing_time_list = []
        n_repeats = config["N_REPEATS"]
        for i in range(n_repeats):
            # Create simulator and run simulation
            sim = DetectionSimulator(sampler=line_sampler,
                                     detector=erx_detector)
            _, seconds_per_line = sim.run(disable_progress_bar=False, save_results=False)

            # Accumulate processing speeds across each iteration
            processing_time_list.append(seconds_per_line)

        # Save processing frequency for current detector
        speed_dict[n_projdims] = processing_time_list
        print(f"Average processing speed: {1 / np.mean(processing_time_list)} lps.")

    # Create local results directory if folder is missing
    if os.path.exists("results") is False:
        os.makedirs("results")

    # Save results after every unique param combination to minimise loss if error occurs
    filename = f"results/speed_results_{n_bands}bands"
    with open(fr"{filename}.pkl", 'wb') as f:
        pickle.dump(speed_dict, f)
    idx += 1


def main():
    # Import test configuration file from local directory
    with open("config_projection.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Run experiments and save numerical results
    get_accuracy_results(config=config)
    if config["RUN_SPEED_TEST"]:
        get_speed_results(config=config)
    return None


if __name__ == '__main__':
    main()
