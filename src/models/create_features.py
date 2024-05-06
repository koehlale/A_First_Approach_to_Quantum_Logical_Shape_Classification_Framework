from typing import Callable

import numpy as np

samples = list[np.ndarray]
dataset = list[samples]

feature_func = Callable[[np.ndarray], np.ndarray]


def init_feature_generator(method: str) -> feature_func:
    """Produce Funktions for the sample generation

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


def produce_features(data: samples, feature_function: feature_func) -> samples:
    features = [feature_function(sample) for sample in data]

    return features
