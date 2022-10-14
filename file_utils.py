import matplotlib.pyplot as plt
import numpy as np
import csv


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


ls = read_numpy_array_from_csv('sample.csv')
print(f'{ls}')
