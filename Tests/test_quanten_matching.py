import numpy as np

from eval import quanten_matching


def test_quanten_k_eq_1():
    k = 1
    X = np.array([[1, 1, 5], [0, 3, 1]])
    Y = np.array([[2, 1, 1], [1, 1, 4]])

    result = np.array([1, 1])

    assert np.allclose(
        quanten_matching(complete_pc_matrices_per_shape_class=[X, Y], k=k), result
    )


def test_quanten_k_eq_2():
    k = 2
    X = np.array([[1, 1, 5], [0, 3, 1]])
    Y = np.array([[2, 1, 1], [1, 1, 4]])

    result = np.array([1, 0])

    assert np.allclose(
        quanten_matching(complete_pc_matrices_per_shape_class=[X, Y], k=k), result
    )


def test_quanten_k_eq_3():
    k = 3
    X = np.array([[1, 1, 5], [0, 3, 1]])
    Y = np.array([[2, 1, 1], [1, 1, 4]])

    result = np.array([0, 1])

    assert np.allclose(
        quanten_matching(complete_pc_matrices_per_shape_class=[X, Y], k=k), result
    )


# def test_pc_computation_x_is_matrix():
#     X = np.array([[1.0, -2.0, 2.0], [3.0, 0.0, 4.0]])

#     result = np.array([[1.0 / 3, -2.0 / 3, 2.0 / 3], [0.6, 0.0, 0.8]])

#     assert np.allclose(norm_matrix(X=X), result)
