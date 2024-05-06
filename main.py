import logging

import matplotlib.pyplot as plt
import numpy as np

from config.conf import Settings
from src.data.raw_data_generator import getting_data
from src.functions.eval import PCA, init_eigenvalue_method, matching
from src.models.create_features import init_feature_generator, produce_features
from src.models.create_normalisation import produce_norma
from src.visualization.plottings import (
    matrix_to_dat_file,
    plot_eigenvalues,
    plot_hitrate_matrix,
    plot_mean_pca_values,
)

samples = list[np.ndarray]
dataset = list[samples]


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
    data_org: dataset = getting_data("lazy")

    # genrating function for feature generation and eigenvalue computation
    feature_generating_function = init_feature_generator(method=cfg.feature_generator)
    eigen_generator = init_eigenvalue_method(cfg.eigenvec_func)
    # normalisation_function = init_normalisation_method(method=cfg.normalizer)

    # i = 0
    # j = 0
    # fig, ax = plt.subplots(8, 10, figsize=(29.7, 21))

    train_data = []
    validation_data = []
    bin_data = []

    logging.info("Generating features...")
    for samp in data_org:
        feature = produce_features(
            data=samp, feature_function=feature_generating_function
        )

        norm_features = produce_norma(features=feature, method=cfg.normalizer)

        min_val = np.min([np.min(el) for el in norm_features])
        max_val = np.max([np.max(el) for el in norm_features])

        logging.debug(f"Minimaler Wert für das Histogramm ist {min_val}.")
        logging.debug(f"Maximaler Wert für das Histogramm ist {max_val}.")

        bins = np.linspace(min_val, max_val, cfg.number_of_bins + 1)

        hist_features = [
            np.histogram(element, bins, density=True)[0] for element in norm_features
        ]

        # Test einer normalisierung vor der PCA
        # hist_features = [element * np.linalg.norm(element) for element in hist_features]

        number_of_train_data = int(
            np.floor((len(samp) * cfg.ratio_of_train_data) / 100)
        )

        logging.debug(
            f"We use {number_of_train_data} out of {len(samp)} samples for training."
        )

        train_data.append(hist_features[:number_of_train_data])
        validation_data.append(hist_features[number_of_train_data:])
        bin_data.append(bins)
    logging.info("    ...done")

    logging.info("Generating Eigenfunctions...")

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
        # logging.info(f"{meanX = }")
        # print(eigVal)
    logging.info("    ...done")

    if False:
        plot_eigenvalues(eigenvalue_data, cfg)

    # if False:
    #     # Plotting der Pricipal compontens and speichern als csv Dateien für Figure 5
    #     # for i in range(15):
    #     sampel_nr = 12  # 0 12
    #     shape = 0  # 1
    #     ref_shape = 1  # 0 1

    #     example = np.array([validation_data[shape][sampel_nr]])

    #     V = eigenvector_data[ref_shape]
    #     mean = mean_value_data[ref_shape]

    #     norm_example = (example - mean) / np.linalg.norm(example - mean)

    #     pca_value = _PCA_analysis(V, norm_example)
    #     logging.info(f"Y = {pca_value.T@pca_value} and {sampel_nr = }")
    #     x = np.array([i + 1 for i in range(10)])
    #     plt.bar(x, pca_value.T[0] ** 2)
    #     # plt.show()
    #     M = np.array([x, pca_value.T[0] ** 2]).T
    #     M.tofile("foo.csv", sep=",", format="%10.5f")
    #     return
    # if False:
    #     # Plotten die Mittleren
    #     plot_mean_pca_values(
    #         data=validation_data,
    #         eigVec_data=eigenvector_data,
    #         mean_value_data=mean_value_data,
    #         cfg=cfg,
    #     )

    hr_matricies = matching(
        data=validation_data,
        eigenbases=eigenvector_data,
        mean_values=mean_value_data,
        k=cfg.number_of_bins,
    )

    # print(np.array(hr_matricies).shape)
    if False:
        plot_hitrate_matrix(data=hr_matricies, cfg=cfg, showing=False)

    if False:
        for k, matrix in enumerate(hr_matricies):
            matrix_to_dat_file(matrix, f"hr_mat_{cfg.name_str}_{k}")
    logging.info("Programm finished! Yeah... :)")


if __name__ == "__main__":
    main()
