#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

import tools

PATH = '../datasets/'
PLOT_FOLDER = '../../../Results/noise/'


def plot_histograms(n_bins):
    pavia_img = io.loadmat(PATH + 'PaviaU/PaviaU.mat')['paviaU']
    dist1 = np.asarray(pavia_img, dtype='float32')
    dist1, _ = tools.HSIData.normalize(dist1, 'standard')
    dist1 = dist1.flatten()

    salinas_img = io.loadmat(PATH + 'Salinas/Salinas_corrected.mat')['salinas_corrected']
    dist2 = np.asarray(salinas_img, dtype='float32')
    dist2, _ = tools.HSIData.normalize(dist2, 'standard')
    dist2 = dist2.flatten()

    indian_img = io.loadmat(PATH + 'IndianPines/Indian_pines_corrected.mat')['indian_pines_corrected']
    dist3 = np.asarray(indian_img, dtype='float32')
    dist3, _ = tools.HSIData.normalize(dist3, 'standard')
    dist3 = dist3.flatten()

    dists = [dist1, dist2, dist3]
    names = ['Pavia University', 'Salinas', 'Indian Pines']

    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True)
    plt.yscale("log")

    # We can set the number of bins with the *bins* keyword argument.
    for i, ax in enumerate(axs):
        ax.set(xlabel='Pixel values', ylabel='Amount of pixels (log)', title=names[i])
        n, bins, patches = ax.hist(dists[i], bins=n_bins)

        # We'll color code by height, but you could use any scalar
        fracs = n / n.max()
        # we need to normalize the data to 0..1 for the full range of the colormap
        norm = colors.Normalize(fracs.min(), fracs.max())
        # Now, we'll loop through our objects and set the color of each accordingly
        for frac, patch in zip(fracs, patches):
            color = plt.cm.viridis(norm(frac))
            patch.set_facecolor(color)

        # plt.savefig(f'{PATH}{data}_{case}_{noise}.png')

    plt.show()


# Main function
def main():
    n_bins = 100
    plot_histograms(n_bins)


if __name__ == '__main__':
    main()
