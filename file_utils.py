import matplotlib.pyplot as plt
import numpy as np


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
