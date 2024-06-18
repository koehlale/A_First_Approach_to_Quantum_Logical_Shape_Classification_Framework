from pathlib import Path

import numpy as np

from config.conf import create_Settings
from config.logger import create_logger
from src.data.raw_data_generator import getting_data
from src.functions.eval import PCA, init_eigenvalue_method, matching
from src.functions.preprocessing import preprocessing
from src.visualization.plottings import plot_eigenvalues, plot_hitrate_matrix

SETTINGS_PATH = "config/settings.env"
DATA_PATH = Path.cwd() / "data" / "all_shapes.mat"

log = create_logger("main")


def main():
    # logging.basicConfig()
    cfg = create_Settings(path=SETTINGS_PATH)
    # TODO: switch different logging level

    cfg.show_config

    # loading data
    data_org: list[list[np.ndarray]] = getting_data(DATA_PATH)

    # genrating function for feature generation and eigenvalue computation

    eigen_generator = init_eigenvalue_method(cfg.eigenvec_func)

    train_data, validation_data, bin_data = preprocessing(data=data_org, cfg=cfg)

    log.info("Compute eigenvalues and eigenfunctions...")
    eigenvector_data = []
    eigenvalue_data = []
    mean_value_data = []
    for hist_samp in train_data:
        eigVal, eigVec, meanX = PCA(
            data=hist_samp, eig_func=eigen_generator, sorting=cfg.sort_ev
        )
        eigenvector_data.append(eigVec)
        mean_value_data.append(meanX)
        eigenvalue_data.append(eigVal)
    log.info("    ...done")

    plot_eigenvalues(eigenvalue_data, cfg)

    hr_matricies = matching(
        data=validation_data,
        eigenbases=eigenvector_data,
        mean_values=mean_value_data,
        k=cfg.number_of_bins,
    )

    plot_hitrate_matrix(data=hr_matricies, cfg=cfg, showing=False)

    # if False:
    #     for k, matrix in enumerate(hr_matricies):
    #         matrix_to_dat_file(matrix, f"hr_mat_{cfg.name_str}_{k}")
    log.info("Programm finished! Yeah... :)")


if __name__ == "__main__":
    main()
