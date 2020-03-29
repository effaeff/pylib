"""Classes and methods for plotting convenience"""

import matplotlib.pyplot as plt


def modify_axis(axs, xtick_label, ytick_label, xoffset, yoffset, fontsize, grid=True):
    """Change properties of plot axis to make more beautiful"""
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(True)
    axs.spines['left'].set_visible(True)
    axs.spines['right'].set_visible(False)
    bottom = 'off'
    left = 'off'
    if xtick_label != '':
        bottom = 'on'
        axs.get_xaxis().tick_bottom()
    if ytick_label != '':
        left = 'on'
        axs.get_yaxis().tick_left()
    axs.tick_params(
        axis="both",
        which="both",
        bottom="on",
        top=False,
        labelbottom=bottom,
        left="on",
        right=False,
        labelleft=left
    )
    if grid:
        axs.grid(True, linewidth=0.4, zorder=0, linestyle='-', which='major')
    if xtick_label != '':
        labels = axs.get_xticklabels()
        labels[xoffset] = xtick_label
        axs.set_xticklabels(labels)
        for label in axs.get_xticklabels():
            label.set_fontsize(fontsize)
    if ytick_label != '':
        labels = axs.get_yticklabels()
        labels[yoffset] = ytick_label
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

