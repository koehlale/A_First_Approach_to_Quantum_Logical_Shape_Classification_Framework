import logging

import numpy as np

from config.conf import Settings
from config.logger import create_logger
from src.functions.eval import get_number_of_train_data
from src.models.create_features import init_feature_generator, produce_features
from src.models.create_histogram import compute_histogram
from src.models.create_normalisation import produce_norma

log = create_logger("preprocessing")


def preprocessing(
    data: list[list[np.ndarray]], cfg: Settings
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    """Realization of Algorithm 1 from the Paper.

    Args:
        data (list[list[np.ndarray]]): _description_
        cfg (Settings): _description_

    Returns:
        tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]: _description_
    """

    # genrating function for feature generation and eigenvalue computation
    feature_generating_function = init_feature_generator(method=cfg.feature_generator)

    train_data = []
    validation_data = []
    bin_data = []

    log.info("Starting Preprocessing...")
    for sample in data:
        # Compute features using implementation of Eq. (19)
        feature = produce_features(
            data=sample, feature_function=feature_generating_function
        )
        # Compute the normalization using implementation of Eq. (22) or (23)
        norm_features = produce_norma(features=feature, method=cfg.normalizer)

        # Compute the min and max values over all normalized features
        hist_features, bins = compute_histogram(
            features=norm_features, k=cfg.number_of_bins
        )

        # Getting the number of train data
        number_of_train_data = get_number_of_train_data(
            length=len(sample), ratio=cfg.ratio_of_train_data
        )

        # Append the results to lists
        train_data.append(hist_features[:number_of_train_data])
        validation_data.append(hist_features[number_of_train_data:])
        bin_data.append(bins)
    log.info("    ...done")
    return train_data, validation_data, bin_data
