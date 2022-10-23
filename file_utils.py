import matplotlib.pyplot as plt
import numpy as np
import csv
import plot_utils


def read_1D_np_array(filename, delim=','):
    """
    This function reads values separated by delim from a file into a 1D array
    :param filename:
    :param delim:
    :return:
    """
    return np.loadtxt(filename, delimiter=delim)


def write_1D_np_array(filename, data, delim=','):
    """
    This method writes a numpy array into a text file
    :param filename:
    :param data:
    :param delim:
    """
    np.savetxt(filename, data, delimiter=delim)


def write_array_to_csv(filename, data, delim=','):
    """
    This method writes an array into a csv file with a specified delimiter
    :param filename:
    :param data:
    :param delim:
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=delim)
        writer.writerows(data)


def read_numpy_array_from_csv(filename, delim=',', data_type=float):
    """
    This method reads a numpy array from a csv file
    :param data_type:
    :param dtype:
    :param filename:
    :param delim:
    :return:
    """
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=delim))
    return np.array(data, dtype=data_type)

###########################################################################################
# following piece of code is used to generate error scatter plot to be used in the paper #
###########################################################################################
# reports_csv = '../csvresults/master_report.csv'
# results = read_numpy_array_from_csv(reports_csv)
# filtered_results = results[:, [1, 4]]
# filtered_results = filtered_results[filtered_results[:, 0] >= 0]
# plot_utils.spike_train_plot(-10 * np.log10(filtered_results[:, 0]), filtered_results[:, 1] / 10,
#                             x_title='error rate in DB', y_title='spike rate as a fraction of Nyquist rate')
# print('done')
