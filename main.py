import logging

import numpy as np

from config.conf import Settings
from src.data.raw_data_generator import getting_data
from src.functions.eval import (
    PCA,
    get_number_of_train_data,
    init_eigenvalue_method,
    matching,
)
from src.models.create_features import init_feature_generator, produce_features
from src.models.create_histogram import compute_histogram
from src.models.create_normalisation import produce_norma
from src.visualization.data_export import matrix_to_dat_file
from src.visualization.plottings import (
    plot_eigenvalues,
    plot_hitrate_matrix,
    plot_mean_pca_values,
)


def main():
    # logging.basicConfig()
    cfg = Settings(_env_file="config/settings.env")

    if cfg.full_inforamtion:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%d.%m.%Y %H:%M:%S",
            level=logging.DEBUG,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%d.%m.%Y %H:%M:%S",
            level=logging.INFO,
        )

    cfg.show_config

    # loading data
    data_org: list[list[np.ndarray]] = getting_data("lazy")

    # genrating function for feature generation and eigenvalue computation
    feature_generating_function = init_feature_generator(method=cfg.feature_generator)
    eigen_generator = init_eigenvalue_method(cfg.eigenvec_func)

    train_data = []
    validation_data = []
    bin_data = []

    logging.info("Starting Preprocessing...")
    for samp in data_org:
        # Compute features using implementation of Eq. (19)
        feature = produce_features(
            data=samp, feature_function=feature_generating_function
        )
        # Compute the normalization using implementation of Eq. (22) or (23)
        norm_features = produce_norma(features=feature, method=cfg.normalizer)

        # Compute the min and max values over all normalized features
        hist_features, bins = compute_histogram(
            features=norm_features, k=cfg.number_of_bins
        )

        # Getting the number of train data
        number_of_train_data = get_number_of_train_data(
            length=len(samp), ratio=cfg.ratio_of_train_data
        )

        # Append the results to lists
        train_data.append(hist_features[:number_of_train_data])
        validation_data.append(hist_features[number_of_train_data:])
        bin_data.append(bins)
    logging.info("    ...done")

    logging.info("Compute eigenvalues and eigenfunctions...")
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
    logging.info("    ...done")

    plot_eigenvalues(eigenvalue_data, cfg)

    hr_matricies = matching(
        data=validation_data,
        eigenbases=eigenvector_data,
        mean_values=mean_value_data,
        k=cfg.number_of_bins,
    )

    plot_hitrate_matrix(data=hr_matricies, cfg=cfg, showing=False)

    if False:
        for k, matrix in enumerate(hr_matricies):
            matrix_to_dat_file(matrix, f"hr_mat_{cfg.name_str}_{k}")
    logging.info("Programm finished! Yeah... :)")


if __name__ == "__main__":
    main()
