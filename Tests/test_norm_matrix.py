import numpy as np

from eval import norm_matrix


def test_pc_computation_x_is_array():

    X = np.array([[1.0, -2.0, 2.0]])

    result = X / 3.0

    assert np.allclose(norm_matrix(X=X), result)


def test_pc_computation_x_is_matrix():
    X = np.array([[1.0, -2.0, 2.0], [3.0, 0.0, 4.0]])

    result = np.array([[1.0 / 3, -2.0 / 3, 2.0 / 3], [0.6, 0.0, 0.8]])

    assert np.allclose(norm_matrix(X=X), result)
