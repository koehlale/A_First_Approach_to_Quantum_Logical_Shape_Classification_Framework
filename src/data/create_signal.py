from typing import Callable

import numpy as np

signal_func = Callable[[np.ndarray], np.ndarray]


def init_signal(
    method: str, min_bin: float, max_bin: float, num_bins: int
) -> signal_func:
    bins = np.linspace(min_bin, max_bin, num_bins + 1)

    def histo(sampels: list[np.ndarray]) -> signal_func:
        return np.histogram(sampels, bins)

    signal_function = {"hist": histo()}

    return signal_function[method]


def produce_signal(sampels: list[np.ndarray], signal_function: signal_func):
    hist, bin_edges = signal_function(sampels)

    pass
