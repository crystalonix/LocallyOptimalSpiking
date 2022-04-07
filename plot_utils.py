import matplotlib.pyplot as plt
import numpy as np


def plot_function(data, xrange=None, x_label=None, y_label=None, title=None):
    """
    This function is used to plot a function of a single variable
    :rtype: object
    :param title:
    :param xrange:
    :param data: values of the function
    :param x_label:
    :param y_label:
    """
    plt.figure()
    if title is not None:
        plt.title(title)
    if xrange is not None:
        plt.plot(data, xrange)
        plt.show()
    else:
        plt.plot(data)
        plt.show()


def plot_functions(funcs, num_cols=1, plot_titles=None, xranges=None, x_labels=None, y_labels=None, x_ticks_on=True,
                   y_ticks_on=False):
    """
    Plots a list of funcs arranged in a grid
    :param x_ticks_on:
    :param y_ticks_on:
    :param plot_titles:
    :param funcs:
    :param num_cols:
    :param xranges:
    :param x_labels:
    :param y_labels:
    """
    num_rows = len(funcs) // num_cols
    num_rows = num_rows + 1 if num_rows * num_cols < len(funcs) else num_rows
    print('see the number of rows: {} and the number of columns:{}'.format(num_rows, num_cols))
    fig, axs = plt.subplots(num_rows, num_cols)
    for i in range(num_rows + 1):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= len(funcs):
                break
            if num_cols == 1:
                axs[i].plot(funcs[i * num_cols + j])
                axs[i].set_title(plot_titles[i])
                if not x_ticks_on:
                    axs[i].set_xticks([])
            else:
                axs[i, j].plot(funcs[i * num_cols + j])
                axs[i, j].set_title(plot_titles[index])
                if not x_ticks_on:
                    axs[i, j].set_xticks([])
    plt.show()


def plot_functions_in_one_plot(x_funcs, y_funcs=None, legends=None, colors=None, xranges=None, x_label=None,
                               y_label=None):
    """
     displays a given list of functions within a single plot
    :param colors:
    :param legends:
    :param x_funcs:
    :param y_funcs:
    :param xranges:
    :param x_label:
    :param y_label:
    """
    for i in range(len(x_funcs)):
        c = None if colors is None else colors[i]
        lb = None if legends is None else legends[i]
        if y_funcs is None:
            if c is not None:
                plt.plot(x_funcs[i], c, label=lb)
            else:
                plt.plot(x_funcs[i], label=lb)
        else:
            if c is not None:
                plt.plot(x_funcs[i], y_funcs[i], c, label=lb)
            else:
                plt.plot(x_funcs[i], y_funcs[i], label=lb)
    plt.legend()
    plt.show()


def plot_matrix(data, xrange=None, yrange=None, x_label=None, y_label=None, title=None):
    plt.figure()
    if title is not None:
        plt.title(title)
    if xrange is not None or yrange is not None:
        pass
    else:
        plt.imshow(data)
        plt.show()


def spike_train_plot(spike_times, spike_indexes, colors=None, size=0.5, title='spike plot'):
    """
    Shows the spike trains in a scatter plot
    :param size:
    :param title:
    :param spike_times:
    :param spike_indexes:
    :param colors:
    """
    plt.figure()
    if colors is not None:
        plt.scatter(spike_times, spike_indexes, s=size, c=colors)
    else:
        plt.scatter(spike_times, spike_indexes, s=size)
    # plt.xlim([0, np.max(spike_indexes)])
    plt.title(title)
    plt.show()


def plot_multiple_spike_trains(spike_times, spike_indexes, num_cols=1, plot_titles=None, size=0.5, xranges=None,
                               colors=None, x_labels=None, y_labels=None, x_ticks_on=True, y_ticks_on=False):
    """
    Plots a list of funcs arranged in a grid
    :param colors:
    :param size:
    :param spike_indexes:
    :param spike_times:
    :param x_ticks_on:
    :param y_ticks_on:
    :param plot_titles:
    :param num_cols:
    :param xranges:
    :param x_labels:
    :param y_labels:
    """
    num_rows = len(spike_times) // num_cols
    num_rows = num_rows + 1 if num_rows * num_cols < len(spike_times) else num_rows
    print('see the number of rows: {} and the number of columns:{}'.format(num_rows, num_cols))
    fig, axs = plt.subplots(num_rows, num_cols)
    for i in range(num_rows + 1):
        for j in range(num_cols):
            index = i * num_cols + j
            if index >= len(spike_times):
                break
            if num_cols == 1:
                axs[i].scatter(spike_times[i * num_cols + j], spike_indexes[i * num_cols + j], s=size, c=colors)
                # .plot(funcs[i * num_cols + j])
                if plot_titles is not None:
                    axs[i].set_title(plot_titles[i])
                if not x_ticks_on:
                    axs[i].set_xticks([])
            else:
                axs[i, j].scatter(spike_times[i * num_cols + j], spike_indexes[i * num_cols + j], s=size, c=colors)
                # axs[i, j].plot(funcs[i * num_cols + j])
                axs[i, j].set_title(plot_titles[index])
                if not x_ticks_on:
                    axs[i, j].set_xticks([])
    plt.show()


def plot_kernel_spike_profile(spike_times, conv_values, z_scores, kernel_projection, kernel_index, plot_titles=None,
                              width=400, club_z_score_threshold=False,
                              xranges=None, colors=None, x_labels=None, y_labels=None, x_ticks_on=True):
    """
    Plots a list of funcs arranged in a grid
    :param club_z_score_threshold:
    :param kernel_projection:
    :param width:
    :param kernel_index:
    :param z_scores:
    :param conv_values:
    :param colors:
    :param spike_times:
    :param x_ticks_on:
    :param plot_titles:
    :param xranges:
    :param x_labels:
    :param y_labels:
    """
    line_width = 1.0
    n = 4 if not club_z_score_threshold else 3
    k = 3 if not club_z_score_threshold else 1
    fig, axs = plt.subplots(n)
    fig.suptitle(f'spike profile for {kernel_index}-th kernel generating total {len(spike_times)} spikes')
    axs[0].plot(conv_values, linewidth=line_width)
    axs[0].set_title('signal kernel convolution')
    axs[1].plot(np.array(z_scores) / (1.0 if not club_z_score_threshold else np.max(z_scores)), linewidth=line_width)
    axs[1].set_title('z-score value for the sliding kernel')
    spike_heights = [1.0 for i in range(len(spike_times))]
    axs[2].bar(spike_times, spike_heights, width=width)
    axs[2].set_title(f'bar plot of spike times for the {kernel_index}-th kernel')
    print(f'check the vals: kernel projection max{np.max(kernel_projection)} '
          f'and max z_score:{np.max(z_scores)} and type: {type(kernel_projection)}')
    axs[k].plot(kernel_projection, linewidth=line_width)
    axs[k].set_title('z score and perpendicular projection of the sliding kernel' if club_z_score_threshold else
                     'Norm of the perpendicular projection of the sliding kernel on the existing span')
    if plot_titles is not None:
        axs[0].set_title(plot_titles[0])
        axs[1].set_title(plot_titles[1])
        axs[2].set_title(plot_titles[2])
    if not x_ticks_on:
        axs[0].set_xticks([])
        axs[1].set_xticks([])
        axs[2].set_xticks([])
    # plt.show()


# # x = [[2, 3, 7], [4, 7, 10], [1, 5, 9]]
# # y = [[1, 2, 3], [2, 3, 5], [4, 5, 6]]
# # plot_functions_in_one_plot(x_funcs=x, y_funcs=y, legends=['first', 'second', 'third'])
# fig, axs = plt.subplots(3)
# axs[0].plot([1, 2, 5, 6, -3, 4, 5, -2])
# axs[1].plot([1, 2, 5, 6, -3, 4, 5, -2])
# spike_times = np.array([2, 4, 7, 2000, 400, 30000])
# axs[2].bar(spike_times, [1.0 for i in range(len(spike_times))], width=20)
# plt.show()
def plot_matrix_as_heatmap(a, title):
    """
    As the name suggests this method displays any given matrix as a heatmap plot
    :param a: matrix
    :param title: title of the plot
    """
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.title = title
    plt.show()
