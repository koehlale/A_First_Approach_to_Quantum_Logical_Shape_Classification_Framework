from typing import Callable

import numpy as np

feature_func = Callable[[np.ndarray], np.ndarray]


def init_feature_generator(method: str) -> feature_func:
    """Produce functions for the sample generation

    Args:
        method (str): _description_

    Returns:
        Callable[[np.ndarray], np.ndarray]: _description_
    """

    def D1(vector: np.ndarray):
        central_point = np.mean(vector, axis=0)
        return np.linalg.norm(vector - central_point, axis=1)

    func = {"D1": D1}

    return func[method]


def produce_features(
    data: list[np.ndarray], feature_function: feature_func
) -> list[np.ndarray]:
    """Takes the samples and the feature_function and evaluates feature_function(samples).

    Args:
        data (list[np.ndarray]): _description_
        feature_function (feature_func): _description_

    Returns:
        samples: _description_
    """
    features = [feature_function(sample) for sample in data]

    return features
