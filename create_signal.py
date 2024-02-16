from typing import Callable

import numpy as np

samples = list[np.ndarray]
dataset = list[samples]

signal_func = Callable[[np.ndarray], np.ndarray]


def init_signal(
    method: str, min_bin: float, max_bin: float, num_bins: int
) -> signal_func:

    bins = np.linspace(min_bin, max_bin, num_bins + 1)

    def histo(samp: samples) -> signal_func:
        return np.histogram(samp, bins)

    signal_function = {"hist": histo()}

    return signal_function[method]


def produce_signal(samp: samples, signal_function: signal_func):

    hist, bin_edges = signal_function(samp)

    pass
