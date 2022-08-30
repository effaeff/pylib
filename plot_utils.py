"""Classes and methods for plotting convenience"""

import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import matplotlib.ticker as mticker

def hist(data, filename, nb_bins='fd', xlabel='Data', fontsize=14, figsize=(7, 4), save=False):
    """Plot histogram of data"""
    __, axs = plt.subplots(1, 1, figsize=figsize)
    bins, __, patches = axs.hist(data, bins=nb_bins)
    # We'll color code by height, but you could use any scalar
    fracs = bins / bins.max()

    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    # Now, we'll loop through our objects and set the color of each accordingly
    for thisfrac, thispatch in zip(fracs, patches):
        cmap = matplotlib.cm.get_cmap("viridis")
        color = cmap(norm(thisfrac))
        thispatch.set_facecolor(color)
    axs.set_xlabel(xlabel, fontsize=fontsize)
    axs.set_ylabel('Distribution', fontsize=fontsize)
    axs.yaxis.set_major_formatter(PercentFormatter(xmax=np.sum(bins)))
    axs.spines.right.set_visible(False)
    axs.spines.top.set_visible(False)

    # kld = 0
    # for idx, __ in enumerate(bins):
        # p_x = bins[idx] / np.sum(bins)
        # if p_x > 0:
            # kld += p_x * math.log(p_x / (1 / len(bins)))
    # print("Kullback-Leibler divergence: {}".format(kld))
    axs.tick_params(axis='both', which='major', labelsize=fontsize-2)
    if save:
        plt.savefig(filename, dpi=600, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
        plt.close()

    # return kld

def modify_axis(axs, xtick_label, ytick_label, xoffset, yoffset, fontsize, grid=True):
    """Change properties of plot axis to make more beautiful"""
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(True)
    axs.spines['left'].set_visible(True)
    axs.spines['right'].set_visible(False)
    bottom = False
    left = False
    if xtick_label != '':
        bottom = True
        axs.get_xaxis().tick_bottom()
    if ytick_label != '':
        left = True
        axs.get_yaxis().tick_left()
    axs.tick_params(
        axis="both",
        which="both",
        bottom=True,
        top=False,
        labelbottom=bottom,
        left=True,
        right=False,
        labelleft=left
    )
    if grid:
        axs.grid(True, linewidth=0.4, zorder=0, linestyle='-', which='major')
    if xtick_label != '':
        labels = axs.get_xticklabels()
        labels[xoffset] = xtick_label
        # axs.xaxis.set_major_locator(mticker.MaxNLocator(max_x))
        ticks_loc = axs.get_xticks().tolist()
        axs.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axs.set_xticklabels(labels)
        for label in axs.get_xticklabels():
            label.set_fontsize(fontsize)
    if ytick_label != '':
        labels = axs.get_yticklabels()
        labels[yoffset] = ytick_label
        # axs.yaxis.set_major_locator(mticker.MaxNLocator(max_y))
        ticks_loc = axs.get_yticks().tolist()
        axs.yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
        axs.set_yticklabels(labels)
        for label in axs.get_yticklabels():
            label.set_fontsize(fontsize)
    return axs

class InteractivePlotter:
    """Class for interactive plotting for optimization tasks"""
    def __init__(self, rows, figsize, fontsize, colors):
        self.plot_idx = 0
        self.fontsize = fontsize
        self.colors = colors
        plt.ion()
        self.fig, self.axs = plt.subplots(rows, 1, figsize=figsize, sharex=True)

        self.plots_pred = [None for __ in range(rows)]
        self.plots_target = [None for __ in range(rows)]

    def init_plot(self, pred, target):
        """Initialize plot"""
        for idx, __ in enumerate(self.plots_target):
            self.plots_target[idx] = self.axs[idx].plot(target[idx], color=self.colors[0])[0]
            self.plots_pred[idx] = self.axs[idx].plot(pred[idx], color=self.colors[1])[0]

        self.axs[-1].set_xlabel('Samples', fontsize=self.fontsize, fontweight='bold')

        plt.tight_layout()
        self.fig.canvas.draw()

    def update_plot(self, pred, target):
        """Update data in axes"""
        for idx, __ in enumerate(self.plots_target):
            self.plots_target[idx].set_ydata(target[idx])
            self.plots_pred[idx].set_ydata(pred[idx])
        for axis in self.axs:
            axis.relim()
            axis.autoscale_view()
        # plt.pause(0.000000001)
        self.fig.canvas.start_event_loop(0.000000001)
        self.fig.canvas.draw()

    def show_plot(self):
        """Show plot at the end of the optimization task with ioff"""
        plt.ioff()
        plt.show()
        plt.close()
