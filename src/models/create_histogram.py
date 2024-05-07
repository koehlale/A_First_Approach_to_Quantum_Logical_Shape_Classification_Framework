import logging

import numpy as np


def _compute_min_max(features: list[np.ndarray]) -> tuple[float, float]:
    min_val = np.min([np.min(el) for el in features])
    max_val = np.max([np.max(el) for el in features])

    logging.debug(f"min | max value for histogram is: {min_val} | {max_val}.")

    return (min_val, max_val)


def _compute_bins(min_max: tuple[float, float], k: int) -> np.ndarray:
    return np.linspace(min_max[0], min_max[1], k + 1)


def compute_histogram(
    features: list[np.ndarray], k: int
) -> tuple[list[np.ndarray], np.ndarray]:
    min_max = _compute_min_max(features=features)

    bins = _compute_bins(min_max=min_max, k=k)

    hist = [np.histogram(element, bins, density=True)[0] for element in features]

    return (hist, bins)
