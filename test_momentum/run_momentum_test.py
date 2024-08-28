import os
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from utils.simulation_core import Sampler, DetectionSimulator
from detectors import erx


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
    line_sampler = Sampler(array=hsi_array, axis=1)
    line_sampler.reset()

    # Iterate over each detector for evaluation
    results_dict = {}
    for momentum in tqdm(config["MOMENTUM_LIST"], disable=True):
        # Indicate current detector being evaluated
        print(f"ERX being evaluated with a momentum of: {momentum}")

        # Configure ERX detector
        erx_detector = erx.ERX(n_bands=n_bands,
                               n_pixels=n_pixels,
                               buffer_len=config["BUFFER_LEN"],
                               momentum=momentum,
                               line_offset=config["LINE_OFFSET"],
                               normalise_md=config["NORMALISE_MD"],
                               threshold=config["THRESHOLD"])

        # Repeat experiment multiple times if specified
        avg_processing_time = 0
        detection_results = None
        n_repeats = config["N_REPEATS"]
        for i in range(n_repeats):
            # Create simulator and run simulation
            sim = DetectionSimulator(sampler=line_sampler,
                                     detector=erx_detector)
            detection_results, seconds_per_line = sim.run(disable_progress_bar=False)

            # Accumulate processing speeds across each iteration
            avg_processing_time += seconds_per_line

        # Average processing time over all experiment iterations
        avg_processing_time /= n_repeats

        # Save processing frequency and results for current detector
        results_dict["time"] = avg_processing_time
        results_dict["momentum"] = momentum
        results_dict["detection_results"] = detection_results

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
