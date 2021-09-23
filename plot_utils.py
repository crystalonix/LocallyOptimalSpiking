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


def plot_functions(funcs, num_cols=1, xranges=None, x_labels=None, y_labels=None):
    """
    Plots a list of funcs arranged in a grid
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
            axs[i, j].plot(funcs[i * num_cols + j])
    plt.show()


def plot_functions_in_one_plot(funcs, xranges=None, x_label=None, y_label=None):
    """
     displays a given list of functions within a single plot
    :param funcs:
    :param xranges:
    :param x_label:
    :param y_label:
    """
    for i in range(len(funcs)):
        plt.plot(funcs[i])
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


r = 7
k1 = 0.5
a1 = 1.2
k2 = 0.5
k3 = 0.5
x = np.arange(0, 0.5, 0.001)
y = (1 - r * x) * (1 - r * x) / (1 - x * x)
plt.plot(x, y)
plt.show()
