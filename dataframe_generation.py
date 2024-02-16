import numpy as np
import pandas as pd


def generating_histogram_dataframe(
    hists: list[list[np.ndarray]], name_list: list[str]
) -> pd.DataFrame:
    # * Hist hat die Dimension: (8, 7000, 10)
    (num_shapes, num_data, num_dim) = np.shape(hists)
    df = pd.DataFrame()
    shape_name_list = []
    for count, hist in enumerate(hists):
        shape_name_list += [name_list[count]] * num_data

        a, b = num_data * count, num_data * (count + 1)
        hist_matrix = np.array(hist)
        # temp_df = pd.DataFrame(hist_matrix, index=[i for i in range(a, b)])
        temp_df = pd.DataFrame(hist_matrix)
        df = pd.concat([df, temp_df])
    df.columns = [f"x_{i+1}" for i in range(num_dim)]
    df["shape"] = shape_name_list

    return df
