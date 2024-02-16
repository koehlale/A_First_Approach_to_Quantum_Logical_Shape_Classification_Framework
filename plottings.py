import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.markers import MarkerStyle

from config.conf import Config, name_str
from eval import PCA_analysis


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
    pass


def plot_eigenvalues(eigVal: list[np.ndarray], cfg: Config) -> None:
    for element in eigVal:
        plt.plot(element)
    plt.yscale("log")
    plt.legend(["3", "4", "5", "6", "7", "8", "9", "0"])
    # plt.show()
    folder = cfg.data_output_folder
    name = f"eigenvalues_{name_str(cfg)}.png"
    plt.savefig(folder + "/" + name)
    _save_notation(folder, name)
    plt.close()


def plot_hitrate_matrix(
    data: list[np.ndarray],
    cfg: Config,
    showing: bool = False,
) -> None:
    logging.info("Creating Plot...")
    n = 3
    m = 3
    fig1, ax = plt.subplots(n, m, figsize=(10, 10))  # ,sharex=True, sharey=True)
    # (21,21)

    i, j = 0, 0
    for k in range(len(data)):
        object = data[k]

        # cearting plot
        ax[i, j].matshow(object, cmap="binary", vmin=0, vmax=1)

        if j != m - 1:
            j += 1
        elif j == m - 1:
            i += 1
            j = 0

    # fig.suptitle(f"{cfg.feature_generator}_{cfg.normalizer}_nr_PV", fontsize=16)
    for axs in ax.flat:
        axs.set(xlabel="x-label", ylabel="y-label")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for axs in ax.flat:
        axs.label_outer()

    folder = cfg.data_output_folder
    name = f"hr_mat_{name_str(cfg)}.png"
    # folder = "output"
    # name = f"pca_{cfg.feature_generator}_{cfg.normalizer}.png"
    fig1.savefig(folder + "/" + name)
    if showing:
        plt.show()
    logging.info("    ...done")
    _save_notation(folder, name)
    plt.close()


def plot_mean_pca_values(
    data: list[np.ndarray],
    eigVec_data: list[np.ndarray],
    mean_value_data: list[np.ndarray],
    cfg: Config,
    showing: bool = False,
) -> None:
    logging.info("Creating Plot...")
    fig, ax = plt.subplots(8, 8, figsize=(29.7, 21))
    for i in range(len(eigVec_data)):
        eigenVec = eigVec_data[i]
        meanX = mean_value_data[i]

        j = 0
        for hist_samp in data:
            proz_value = PCA_analysis(eigenVec, hist_samp - meanX)
            mean_proz_value = np.mean(proz_value, axis=1)

            # cearting plot
            x = [i for i in range(1, cfg.number_of_bins + 1)]
            ax[i, j].bar(x, mean_proz_value, width=0.5, color="black")
            j += 1

    # fig.suptitle(
    #     f"pca_proc_value_{cfg.feature_generator}_{cfg.normalizer}", fontsize=16
    # )
    fig.savefig(f"pca_{cfg.feature_generator}_{cfg.normalizer}.png")
    if showing:
        plt.show()
    logging.info("    ... done")
    pass


def plot_histogram_prob(
    hist: list[np.ndarray],
    eigVec: list[np.ndarray],
    mean_val: list[np.ndarray],
    main_ax: int,
) -> None:
    # * Hist hat die Dimension: (8, 7000, 10)
    # * eigVec hat die Dimension: (8,10,10)
    # * mean_val hat die Dimension: (8,10)

    logging.info("Creating Plot...")
    plt.close()
    fig, ax = plt.subplots(3, 3, figsize=(15, 15))

    def point_calc(m, v, i):
        return [m[i] - 0.5 * v[i], m[i] + 0.5 * v[i]]

    # m, n, l = np.shape(hist)
    # # hists_from_shape = np.array(hist[0])
    # # x = hists_from_shape[:, 0]
    # # print(len(x))
    # # print(f"{m = }; {n = }; {l = }")

    poss_axes = [i for i in range(10)]
    poss_axes.pop(main_ax)

    hist1 = np.array(hist[0])
    hist2 = np.array(hist[1])
    hist3 = np.array(hist[7])

    mean1 = mean_val[0]
    mean2 = mean_val[1]
    mean3 = mean_val[7]

    vec1 = eigVec[0]
    vec2 = eigVec[1]
    vec3 = eigVec[7]

    i, j = 0, 0
    for k in poss_axes:
        x1 = hist1[:, main_ax]
        y1 = hist1[:, k]
        x2 = hist2[:, main_ax]
        y2 = hist2[:, k]
        x3 = hist3[:, main_ax]
        y3 = hist3[:, k]

        m1 = mean1[[main_ax, k]]
        m2 = mean2[[main_ax, k]]
        m3 = mean3[[main_ax, k]]

        v1 = vec1.T[0][[main_ax, k]]
        v2 = vec2.T[0][[main_ax, k]]
        v3 = vec3.T[0][[main_ax, k]]

        # Plotte die Punkte
        ax[i][j].scatter(x2, y2, marker=MarkerStyle("s", fillstyle="none"), color="C1")
        ax[i][j].scatter(x1, y1, marker=MarkerStyle("^", fillstyle="none"), color="C0")
        ax[i][j].scatter(x3, y3, marker=MarkerStyle("o", fillstyle="none"), color="C2")

        # ## Plotte die Pfeile
        # xmax = np.max([x1, x2, x3])
        # xmin = np.min([x1, x2, x3])
        # ymax = np.max([y1, y2, y3])
        # ymin = np.min([y1, y2, y3])

        ax[i][j].plot(
            point_calc(m1, v1, 0),
            point_calc(m1, v1, 1),
            color="blue",
            linewidth=2,
        )
        ax[i][j].plot(
            point_calc(m2, v2, 0),
            point_calc(m2, v2, 1),
            color="red",
            linewidth=2,
        )
        ax[i][j].plot(
            point_calc(m3, v3, 0),
            point_calc(m3, v3, 1),
            color="black",
            linewidth=2,
        )

        # Setze die Axen-Labels
        ax[i][j].set_xlabel(f"x_{main_ax+1}")
        ax[i][j].set_ylabel(f"x_{k+1}")

        if j != 2:
            j += 1
        elif j == 2:
            i += 1
            j = 0
    plt.show()
    # fig.savefig(f"hist_dist_main_{main_ax+1}.png")
    logging.info("    ... done")

    # fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    # i, j = 0, 0
    # for k in range(len(hist)):
    #     object = hist[k]

    #     # cearting plot
    #     ax[i, j].scatter(hist[k])

    #     # Zeilen / Spalten durchlauf
    #     if j != 2:
    #         j += 1
    #     elif j == 2:
    #         i += 1
    #         j = 0

    # # fig.suptitle(f"{cfg.feature_generator}_{cfg.normalizer}_nr_PV", fontsize=16)
    # for axs in ax.flat:
    #     axs.set(xlabel="x-label", ylabel="y-label")

    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # for axs in ax.flat:
    #     axs.label_outer()

    pass
    pass
