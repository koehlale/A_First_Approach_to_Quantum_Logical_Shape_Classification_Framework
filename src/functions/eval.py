import logging
from typing import Callable

import numpy as np
from sklearn.preprocessing import normalize

samples = list[np.ndarray]


def init_eigenvalue_method(
    method: str,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    func = {"eig": np.linalg.eig, "eigh": np.linalg.eigh}
    return func[method]


def PCA(
    data: samples,
    eig_func: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    sorting: bool,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(data)
    mX = np.mean(X, axis=0)
    X = X - mX
    E = np.cov(X.T)

    logging.debug("--- PCA-Funktion---")
    logging.debug(f"Dimension der Matrix ={E.shape}")
    logging.debug(f"E = {E}")

    eigVal, eigVec = eig_func(E)
    logging.debug(f"Eigenvektoren = {eigVec} and {type(eigVec)}")
    logging.debug(f"Eigenwerte = {eigVal}")
    # print(f"{np.round(eigVec @ eigVec.T, 3)}")
    if sorting:
        idx = eigVal.argsort()[::-1]
        # idx = eigVal.argsort()
        return eigVal[idx], eigVec[idx], mX
    return eigVal, eigVec, mX


def _PCA_analysis(EigenVectors: np.ndarray, norm_vector: np.ndarray) -> np.ndarray:
    # Point_Matrix has shape of (NumberData, NumberOfFeatures)

    Principle_Values = EigenVectors.T @ norm_vector.T

    logging.debug("--- PCA-Analysis ---")
    logging.debug(f"Dimension der Hauptkomponten = {Principle_Values.shape}")
    logging.debug(f"Y = {Principle_Values} ")

    return Principle_Values


def PCA_analysis(EigenVectors: np.ndarray, points: list[np.ndarray]) -> np.ndarray:
    Point_Matrix = np.array(points)

    Principle_Values = _PCA_analysis(EigenVectors, Point_Matrix)

    total = np.sum(np.abs(Principle_Values), axis=0)
    logging.debug("--- PCA-Analysis ---")
    logging.debug(f"Dimension der Hauptkomponten = {Principle_Values.shape}")
    logging.debug(f"Y = {Principle_Values} ")
    logging.debug(f"Dimension von Total = {np.shape(total)}")

    return (np.abs(Principle_Values) / total) * 100


def pc_computation(V: np.matrix, X: np.matrix) -> np.matrix:
    """Computes the principal components of a row structed matrix X and a row structered eigenspace. The resulting matrix is also row structered

    row structured: A = [ a11 a12 ; a21 a22 ] and A[0] = [a11 a12]

    Args:
        V (np.matrix): expecting V be a row structured matrix
        X (np.matrix): expecting X be a row structured matrix

    Returns:
        np.matrix: result is also row structured
    """

    # return X @ V.T
    return X @ V


def norm_matrix(X: np.matrix) -> np.matrix:
    return normalize(X=X, axis=1, norm="l2")


def data_preprocess(
    data: samples, eigenbases: list[np.ndarray], mean_values: list[np.ndarray]
) -> list[list[np.matrix]]:
    logging.info("Compute preprocessing for matching...")
    all_pc_matrices = []
    for shape_class_data in data:
        pc_matrices_per_class = []
        for eigVec, mean in zip(eigenbases, mean_values):
            norm_samples = norm_matrix(shape_class_data - mean)
            pc_matrix = pc_computation(V=eigVec, X=norm_samples)
            pc_matrices_per_class.append(pc_matrix)
        all_pc_matrices.append(pc_matrices_per_class)
    logging.info("    ...done")
    return all_pc_matrices


def _prob_quanten_matching(
    complete_pc_matrices_per_shape_class: list[np.matrix], k: int
) -> np.ndarray:
    n = len(complete_pc_matrices_per_shape_class)
    all_probs = []
    for i in range(n):
        pc_matrix = complete_pc_matrices_per_shape_class[i]
        squared_pc_matrix = pc_matrix**2  # 3000 row; 10 col
        prob = np.sum(squared_pc_matrix[:, :k], axis=1)
        all_probs.append(prob)
    return np.array(all_probs)


def quanten_matching(
    complete_pc_matrices_per_shape_class: list[np.matrix], k: int
) -> np.ndarray:
    probability_matrix = _prob_quanten_matching(complete_pc_matrices_per_shape_class, k)

    _check_for_multiple_entries(probability_matrix)

    return np.argmax(
        probability_matrix, axis=0
    )  # Gibt Index des größten Wertes in einer Spalte


def quanten_matching_prob_mat(
    complete_pc_matrices_per_shape_class: list[np.matrix], k: int
) -> np.ndarray:
    probability_matrix = _prob_quanten_matching(complete_pc_matrices_per_shape_class, k)
    return np.mean(probability_matrix, axis=1)


def _check_for_square_sum_eq_one(matrix: np.ndarray, tol: float = 1e-8) -> None:
    for vec in matrix:
        sum = np.sum(vec)
        if np.abs(1 - sum) > tol:
            print(sum)
            ...


def _check_for_multiple_entries(matrix: np.ndarray) -> None:
    for vec in matrix.T:
        _, arr = np.unique(vec, return_counts=True)
        if np.max(arr) > 1:
            logging.info(
                f"We have multiple times the same probability.\nThis is mybe a problem. \nSee: {_}"
            )


def matching(
    data: samples,
    eigenbases: list[np.ndarray],
    mean_values: list[np.ndarray],
    k=int,
    plotting_hr: bool = False,
) -> list[np.ndarray]:
    num_shapes, num_samples, _ = np.array(data).shape
    logging.debug(f"{num_shapes = } , {num_samples = }")

    pc_matrices = data_preprocess(
        data=data, eigenbases=eigenbases, mean_values=mean_values
    )

    logging.info("Compute matching...")
    count = 1
    hr_matrices = []
    for k0 in range(1, k):
        hit_rate_matrix = np.zeros((num_shapes, num_shapes))
        for shape_class in range(num_shapes):
            best_matching = quanten_matching(pc_matrices[shape_class], k0)
            logging.debug(f"{count}: {best_matching = } --- {best_matching.shape}")
            # print(f"{count}: {best_matching = } --- {best_matching.shape}")
            count += 1
            # hit_rate_matrix[shape_class] = best_matching
            for matching_res in range(num_shapes):
                hit_rate_matrix[shape_class, matching_res] = np.sum(
                    [best_matching == matching_res]
                ) / len(best_matching)
        hr_matrices.append(hit_rate_matrix)
        if plotting_hr:
            np.set_printoptions(precision=3, suppress=True)
            print(hit_rate_matrix)
    logging.info("    ...done")
    return hr_matrices
