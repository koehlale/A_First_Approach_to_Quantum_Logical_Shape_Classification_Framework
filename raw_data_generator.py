import logging
from pathlib import Path

import numpy as np
import scipy.io as scio


def getting_data(method: str) -> list[list[np.ndarray]]:

    possible_methods = ["lazy"]

    if method not in possible_methods:
        raise NotImplementedError()

    path = Path.cwd() / "Data" / "all_shapes.mat"
    return read_data(path)


def read_data(path: Path) -> list[list[np.ndarray]]:
    logging.info(f"reading Data...")

    data = []

    Matlab_Cell: np.ndarray = scio.loadmat(path)["shapes"]

    for object_class in Matlab_Cell:
        objects = []
        for object in object_class:
            objects.append(np.array(object, dtype="float"))
        data.append(objects)
    logging.info(f"    ...done")
    return data
