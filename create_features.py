from typing import Callable, Optional

import numpy as np

samples = list[np.ndarray]
dataset = list[samples]

feature_func = Callable[[np.ndarray], np.ndarray]


def init_feature_generator(method: str, N: Optional[int]) -> feature_func:
    """Produce Funktions for the sample generation

    Args:
        method (str): _description_
        N (Optional[int]): _description_

    Returns:
        Callable[[np.ndarray], np.ndarray]: _description_
    """

    def D1(vector: np.ndarray):
        central_point = np.mean(vector, axis=0)
        return np.linalg.norm(vector - central_point, axis=1)

    def D1R(vector: np.ndarray, n: int = N, rand_seed: int = N):
        np.random.seed(rand_seed)
        central_point = np.mean(vector, axis=0)
        index = np.random.randint(len(vector), size=n)
        return np.linalg.norm(vector[index] - central_point, axis=1)

    def D2(vector: np.ndarray, n: int = N, rand_seed: int = N):
        np.random.seed(rand_seed)
        len_vec = len(vector)
        index1 = np.random.randint(len_vec, size=n)
        index2 = np.random.randint(len_vec, size=n)
        eq: np.ndarray = index1 == index2
        while eq.any():
            index2[eq] = np.random.randint(len_vec, size=len(index2[eq]))
            eq = index1 == index2

        return np.linalg.norm(vector[index1] - vector[index2], axis=1)

    func = {"D1": D1, "D1R": D1R, "D2": D2}

    return func[method]


def produce_features(data: samples, feature_function: feature_func) -> samples:

    features = [feature_function(sample) for sample in data]

    return features
