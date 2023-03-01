# Functions used in to plot avg and std of several runs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use('tableau-colorblind10')


def mean_std(data):
    """
    Take average and std out of several runs
    :param data: np.array(runs, max_epoch), data from several runs
    :return: (np.array(max_epoch), np.array(max_epoch)), average and std epoch-wise
    """

    return np.mean(data, axis=0).flatten(), np.std(data, axis=0).flatten()


def plot_avg_std(data, label):
    """
    Plot the line of average and shaded region of one standard deviation
    :param data: np.array(runs, max_epoch), data from several runs
    :param label: string, name of the plotted line to appear in the legend
    :return: -
    """
    avg, std = mean_std(data)

    # prevent negative bars on a log scale
    l = 0.4
    avg_bot = avg - std
    avg_bot[std >= l * avg] = avg[std >= l * avg] - l * std[std >= l * avg]

    plt.plot(avg, linestyle="-", label=label)
    plt.fill_between(np.arange(0, len(avg)),
                     avg_bot, avg + std,
                     alpha=0.2)
