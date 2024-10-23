import os
import yaml
import pickle
import itertools
import numpy as np
from tqdm import tqdm
from utils.simulation_core import Sampler, DetectionSimulator
from detectors import rx_baseline, rt_ck_rxd, rx_bil, cdlss_ad, lbl_ad, erx


def get_speed_results(config: dict):
    """
    :param: config: dictionary with configuration file inputs
    :return:
    """

    # Perform grid search across all dataset parameter combinations
    idx = 0
    for n_pixels, n_lines, n_bands in itertools.product(config["N_PIXELS"], config["N_LINES"], config["N_BANDS"]):
        # Store current params
        params_dict = {"n_pixels": n_pixels,
                       "n_lines": n_lines,
                       "n_bands": n_bands}

        # Create hsi line sampler from dummy dataset (only use current number of pixels, lines and bands)
        line_sampler = Sampler(array=(np.random.rand(n_pixels, n_lines, n_bands) * 10000).astype(int),
                               axis=1)
        line_sampler.reset()

        # Define detectors and store in dictionary
        detector_dict = {}

        # Configure RX Baseline detector
        if config["rx_baseline"]["include"]:
            rx_baseline_detector = rx_baseline.RXBaseline(n_bands=n_bands,
                                                          n_pixels=n_pixels,
                                                          buffer_len=config["rx_baseline"]["buffer_len"],
                                                          line_offset=config["rx_baseline"]["buffer_len"] // 2,
                                                          normalise_md=config["rx_baseline"]["normalise_md"])
            detector_dict["rx_baseline"] = rx_baseline_detector

        # Configure RT-CK-RXD detector
        if config["rt_ck_rxd"]["include"]:
            rt_ck_rxd_detector = rt_ck_rxd.RT_CK_RXD(n_bands=n_bands,
                                                     n_pixels=n_pixels,
                                                     buffer_len=config["rt_ck_rxd"]["buffer_len"],
                                                     normalise_md=config["rt_ck_rxd"]["normalise_md"])
            detector_dict["rt_ck_rxd"] = rt_ck_rxd_detector

        # Configure RX-BIL detector
        if config["rx_bil"]["include"]:
            rx_bil_detector = rx_bil.RX_BIL(n_bands=n_bands,
                                            n_pixels=n_pixels,
                                            buffer_len=config["rx_bil"]["buffer_len"],
                                            pixel_dropout=config["rx_bil"]["pixel_dropout"],
                                            normalise_md=config["rx_bil"]["normalise_md"])
            detector_dict["rx_bil"] = rx_bil_detector

        # Configure CDLSS-AD detector
        if config["cdlss_ad"]["include"]:
            cdlss_ad_detector = cdlss_ad.CDLSS_AD(n_bands=n_bands,
                                                  n_pixels=n_pixels,
                                                  buffer_len=config["cdlss_ad"]["buffer_len"],
                                                  normalise_md=config["cdlss_ad"]["normalise_md"])
            detector_dict["cdlss_ad"] = cdlss_ad_detector

        # Configure LBL-AD detector
        if config["lbl_ad"]["include"]:
            lbl_ad_detector = lbl_ad.LblAD(n_bands=n_bands,
                                           n_pixels=n_pixels,
                                           buffer_len=config["lbl_ad"]["buffer_len"],
                                           pca_dims=config["lbl_ad"]["pca_dims"],
                                           normalise_md=config["lbl_ad"]["normalise_md"])
            detector_dict["lbl_ad"] = lbl_ad_detector

        # Configure ERX detector
        if config["erx"]["include"]:
            erx_detector = erx.ERX(n_bands=n_bands,
                                   n_pixels=n_pixels,
                                   buffer_len=config["erx"]["buffer_len"],
                                   n_projdims=config["erx"]["projected_dimensions"],
                                   momentum=config["erx"]["momentum"],
                                   normalise_md=config["erx"]["normalise_md"])
            detector_dict["erx"] = erx_detector

        # Iterate over each detector for evaluation
        speed_dict = {}
        print(f"Detectors being evaluated for {n_pixels} pixels, {n_lines} lines, and {n_bands} bands.")
        for model_name, detector in tqdm(detector_dict.items(), disable=True):
            # Repeat experiment multiple times if specified
            processing_time_list = []
            n_repeats = config["N_REPEATS"]
            print(f"Evaluating {model_name}...")
            for i in range(n_repeats):
                # Create simulator and run simulation
                sim = DetectionSimulator(sampler=line_sampler,
                                         detector=detector)
                _, seconds_per_line = sim.run(disable_progress_bar=False, save_results=False)

                # Accumulate processing speeds across each iteration
                processing_time_list.append(seconds_per_line)

            # Save processing frequency for current detector
            speed_dict[model_name] = processing_time_list
            print(f"Average processing speed: {1 / np.mean(processing_time_list)} lps.")

        # Include model params so speed results are identifiable to dataset size
        results_dict = {"params": params_dict,
                        "time": speed_dict}

        # Create local results directory if folder is missing
        if os.path.exists("results") is False:
            os.makedirs("results")

        # Save results after every unique param combination to minimise loss if error occurs
        filename = f"results/results_{n_pixels}pixels_{n_lines}lines_{n_bands}bands"
        with open(fr"{filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)
        idx += 1


def main():
    # Import test configuration file from local directory
    with open("config_speed.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Run experiments and save numerical results
    get_speed_results(config=config)
    return None


if __name__ == '__main__':
    main()
