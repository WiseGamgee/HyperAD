import os
import yaml
import pickle
import numpy as np
from tqdm import tqdm
from utils.simulation_core import Sampler, DetectionSimulator
from detectors import rx_baseline, rt_ck_rxd, rx_bil, cdlss_ad, lbl_ad, erx


#
def get_threshold_results(config: dict):
    """
    :param config: configuration dictionary imported from yaml
    :return:
    """
    # Load hsi dataset from file
    hsi_dir = config["HSI_DIR"]
    hsi_array = np.load(f"../{hsi_dir}")
    n_pixels, n_lines, n_bands = hsi_array.shape

    # Create hsi line sampler from dummy dataset
    line_sampler = Sampler(array=hsi_array, axis=1, reverse=config["REVERSE"])
    line_sampler.reset()

    # Define detectors and store in dictionary
    detector_dict = {}
    if config["rx_baseline"]["include"]:
        rx_baseline_detector = rx_baseline.RXBaseline(n_bands=n_bands,
                                                      n_pixels=n_pixels,
                                                      buffer_len=config["rx_baseline"]["buffer_len"],
                                                      line_offset=config["rx_baseline"]["buffer_len"] // 2,
                                                      normalise_md=config["rx_baseline"]["normalise_md"])
        detector_dict["rx_baseline"] = rx_baseline_detector

    if config["rt_ck_rxd"]["include"]:
        rt_ck_rxd_detector = rt_ck_rxd.RT_CK_RXD(n_bands=n_bands,
                                                 n_pixels=n_pixels,
                                                 buffer_len=config["rt_ck_rxd"]["buffer_len"],
                                                 normalise_md=config["rt_ck_rxd"]["normalise_md"])
        detector_dict["rt_ck_rxd"] = rt_ck_rxd_detector

    if config["rx_bil"]["include"]:
        rx_bil_detector = rx_bil.RX_BIL(n_bands=n_bands,
                                        n_pixels=n_pixels,
                                        buffer_len=config["rx_bil"]["buffer_len"],
                                        pixel_dropout=config["rx_bil"]["pixel_dropout"],
                                        normalise_md=config["rx_bil"]["normalise_md"])
        detector_dict["rx_bil"] = rx_bil_detector

    if config["cdlss_ad"]["include"]:
        cdlss_ad_detector = cdlss_ad.CDLSS_AD(n_bands=n_bands,
                                              n_pixels=n_pixels,
                                              buffer_len=config["cdlss_ad"]["buffer_len"],
                                              normalise_md=config["cdlss_ad"]["normalise_md"])
        detector_dict["cdlss_ad"] = cdlss_ad_detector

    if config["lbl_ad"]["include"]:
        lbl_ad_detector = lbl_ad.LblAD(n_bands=n_bands,
                                       n_pixels=n_pixels,
                                       buffer_len=config["lbl_ad"]["buffer_len"],
                                       pca_dims=config["lbl_ad"]["pca_dims"],
                                       normalise_md=config["lbl_ad"]["normalise_md"])
        detector_dict["lbl_ad"] = lbl_ad_detector

    if config["erx"]["include"]:
        erx_detector = erx.ERX(n_bands=n_bands,
                               n_pixels=n_pixels,
                               n_projdims=config["erx"]["projected_dimensions"],
                               buffer_len=config["erx"]["buffer_len"],
                               momentum=config["erx"]["momentum"],
                               line_offset=config["erx"]["line_offset"],
                               normalise_md=config["erx"]["normalise_md"],
                               threshold=config["erx"]["threshold"])
        detector_dict["erx"] = erx_detector

    # Iterate over each detector for evaluation
    for name, detector in tqdm(detector_dict.items(), disable=True):
        # Iterate over all threshold values
        results_dict = {}
        print(f"{name} being evaluated.")

        # Repeat experiment multiple times if specified
        processing_time_list = []
        detection_results_list = []
        n_repeats = config["N_REPEATS"]
        for i in range(n_repeats):
            # Create simulator and run simulation
            sim = DetectionSimulator(sampler=line_sampler,
                                     detector=detector)
            md_map, seconds_per_line = sim.run(disable_progress_bar=False)

            # Accumulate processing speeds across each iteration
            processing_time_list.append(seconds_per_line)
            detection_results_list.append(md_map)

        # Save processing times and md results for current detector
        results_dict["times"] = processing_time_list
        results_dict["md_maps"] = detection_results_list

        # Create local results directory if folder is missing
        if os.path.exists("results/") is False:
            os.makedirs("results/")

        # Save results after every model to minimise loss if error occurs
        filename = f"results/threshold_results_{name}"
        with open(fr"{filename}.pkl", 'wb') as f:
            pickle.dump(results_dict, f)


def main():
    # Import test configuration file from local directory
    with open("config_threshold.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Run experiments and save numerical results
    get_threshold_results(config=config)
    return None


if __name__ == '__main__':
    main()
