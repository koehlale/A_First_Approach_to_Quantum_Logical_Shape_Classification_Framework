import logging

import numpy as np


def _save_notation(folder: str, name: str) -> None:
    logging.info(f"A plot was saved: \n Folder: {folder}\n Name:   {name}")


def matrix_to_dat_file(matrix: np.ndarray, name: str) -> None:
    m, n = matrix.shape
    fullname = name + ".dat"
    with open(fullname, "w") as file:
        file.write("x y val\n")
        for i in range(m):
            for j in range(n):
                file.write(f"{i+1} {j+1} {matrix[i,j]}\n")
        file.write(" \n")
