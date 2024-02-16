import numpy as np

from eval import pc_computation


def test_pc_computation_x_is_array():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([2.0, 4.0, 1.0])
    v3 = np.array([1.0, 0.0, 2.0])
    V = np.array([v1, v2, v3])
    X = np.array([[1.0, -2.0, 2.0]])

    result = np.array([[3.0, -4.0, 5.0]])

    assert np.allclose(pc_computation(V=V.T, X=X), result)


def test_pc_computation_x_is_matrix():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([2.0, 4.0, 1.0])
    v3 = np.array([1.0, 0.0, 2.0])
    V = np.array([v1, v2, v3])
    X = np.array([[1.0, -2.0, 2.0], [1.0, 0.0, 4.0]])

    result = np.array([[3.0, -4.0, 5.0], [13.0, 6.0, 9.0]])

    assert np.allclose(pc_computation(V=V.T, X=X), result)
