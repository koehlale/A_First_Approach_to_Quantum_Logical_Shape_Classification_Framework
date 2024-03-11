from typing import Callable

import numpy as np

samples = list[np.ndarray]
dataset = list[samples]

norm_func = Callable[[np.ndarray, float], float]


def init_normalisation_method(method: str) -> norm_func:
    normalizer_function = {"mean": np.mean, "max": np.max}
    return normalizer_function[method]


################################################################################

norma_gen_func = Callable[[np.ndarray, np.ndarray], np.ndarray]


def init_method(method: str) -> norma_gen_func:
    def mean_func(sample: np.ndarray, ref_sample: np.ndarray) -> np.ndarray:
        old_value = np.mean(sample)
        ref_value = np.mean(ref_sample)
        if old_value == 0:
            raise ZeroDivisionError(
                "The mean value of this sample is zero and therefore you would divide with zero in further calculations."
            )
        factor = ref_value / old_value
        return factor * sample

    def max_func(sample: np.ndarray, ref_sample: np.ndarray) -> np.ndarray:
        old_value = np.max(sample)
        ref_value = np.max(ref_sample)
        if old_value == 0:
            raise ZeroDivisionError(
                "The max value of this sample is zero and therefore you would divide with zero in further calculations."
            )
        factor = ref_value / old_value
        return factor * sample

    def zero_one_func(sample: np.ndarray, ref_sample: np.ndarray) -> np.ndarray:
        sample -= np.min(sample)
        max_value = np.max(sample)
        if max_value == 0:
            raise ZeroDivisionError(
                "The max value of this sample is zero and therefore you would divide with zero in further calculations."
            )
        return sample / max_value

    def zero_less_one_func(sample: np.ndarray, max_value: float) -> np.ndarray:
        sample -= np.min(sample)
        if max_value == 0:
            raise ZeroDivisionError(
                "The max value of this sample is zero and therefore you would divide with zero in further calculations."
            )
        return sample / max_value

    poss_options = {
        "mean": mean_func,
        "max": max_func,
        "zeroone": zero_one_func,
        # "0gmax": zero_less_one_func,
    }

    return poss_options[method]


################################################################################


def produce_normalisation(
    features: samples, normalisation_function: norm_func, ref_feature: np.ndarray
) -> samples:
    normalised_features = []
    ref_value = normalisation_function(ref_feature)
    for feature_sample in features:
        old_value = normalisation_function(feature_sample)
        if old_value == 0:
            raise ZeroDivisionError(
                "The normalisation value is zero for the current feature. So it would leed to devision with zero."
            )
        factor = ref_value / old_value
        normalised_features.append(factor * feature_sample)
    return normalised_features


def produce_norma(features: samples, method: str) -> samples:
    func = init_method(method)
    ref_feature = features[0]

    return [func(samp, ref_feature) for samp in features]
