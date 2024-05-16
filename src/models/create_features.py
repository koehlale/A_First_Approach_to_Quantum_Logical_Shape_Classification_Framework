from typing import Callable

import numpy as np

from config.logger import create_logger

feature_func = Callable[[np.ndarray], np.ndarray]

log = create_logger("feature_generation")


def init_feature_generator(method: str) -> feature_func:
    """Produce functions for the sample generation.
    The functions are defined within this function. A reference to this function is stored in the dictionary 'func', with the method string as key in the dictionary.

    Args:
        method (str): A string that refer to the functions defined within this function.

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that takes the matrix of the point cloud as a 'ndarrray' and gives a vector back.
    """

    def D1(vector: np.ndarray):
        central_point = np.mean(vector, axis=0)
        return np.linalg.norm(vector - central_point, axis=1)

    func = {"D1": D1}

    return func[method]


def produce_features(
    data: list[np.ndarray], feature_function: feature_func
) -> list[np.ndarray]:
    """Takes the samples (list[np.ndarray], list of matricies from the pointcloud), and the genreated feature_function and evaluetes the feature of these matrixes.
    Therfore, we take each matrix from the list compute the feature and store the vector in a list again.

    Args:
        data (list[np.ndarray]): A list of matrices that contains the the points clouds.
        feature_function (Callable[[np.ndarray], np.ndarray]): A function that is generated with the 'init_feature_generator' function.

    Returns:
        list[np.ndarry]: The list of feature vectors.
    """
    features = [feature_function(sample) for sample in data]

    return features
