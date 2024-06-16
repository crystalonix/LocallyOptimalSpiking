#
# Copyright (c) 2024 Anik Chattopadhyay, Arunava Banerjee
#
# Author: Arunava Banerjee
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Note: This project is also subject to a provisional patent. The Creative Commons license
# applies to the documentation and code provided herein, but does not grant any rights to
# the patented invention.
#

import numpy as np
import math
import file_utils
import configuration
import plot_utils
import common_utils
import signal_utils

# declare some necessary global param
COEFFICIENTS_FOLDER_PATH = 'kernel_coeffs_5000/'
upsample_factor = 10
sampling_rate = 44100.0 * upsample_factor
length_b = 0
length = 0
# const is used to scale up the values of the gammatone
const = 1e6
audio_filepath = 'audio_text/'
filename = 'frequencies_5000.txt'
coffes_filename = 'solution_{}.txt'
total_available_kernels = 5000


# initialize the step length of bspline and the length of the gammatone filter
def initialize(frequency, bandwidth=0):
    """
    This method initializes the length params of a gamatone
    :param frequency:
    :param bandwidth:
    """
    global length, length_b
    length = int(
        math.ceil((22.0 / frequency) * (1.0 / (configuration.NUMBER_OF_BSPLINE_COMPONENTS + 2.0)) * sampling_rate))
    length_b = length * (configuration.NUMBER_OF_BSPLINE_COMPONENTS + 2)


# calculate the value of the gammatone itself
def calculate_b(frequency, bandwidth, normalize=False):
    """
    This method returns a gammatone filter of a particular frequency and bandwidth
    :param normalize:
    :param frequency:
    :param bandwidth:
    :return:
    """
    b = np.zeros(length_b)
    for i in range(length_b):  # (i=0;i<length_b;i++)
        time = (i / sampling_rate)
        b[i] = time * time * time * np.exp(-2.0 * np.pi * bandwidth * time) * np.cos(
            2.0 * np.pi * frequency * time) * const
    if normalize:
        b = signal_utils.normalize_signal(b)
    return b


def get_gammatone_kernel(frequency, bandwidth):
    """
    Helper method to get the gammtone filter computed by the above method
    :param frequency:
    :param bandwidth:
    :return:
    """
    initialize(frequency, bandwidth)
    return calculate_b(frequency, bandwidth)


def get_basic_bspline(unit_length):
    """
    This method returns a simple bspline function of a given length
    :param unit_length:
    :return:
    """
    # the column contains a basic bspline function
    column = np.zeros(3 * unit_length)
    # now build 2nd order b-spline in column
    for j in range(length):
        _time = j / length
        column[j] = 0.5 * _time * _time
    for j in range(length, 2 * length):
        _time = j / length
        column[j] = 0.5 * (-3.0 + (6.0 * _time) - (2.0 * _time * _time))
    for j in range(2 * length, 3 * length):
        _time = j / length
        column[j] = 0.5 * (3.0 - _time) * (3.0 - _time)
    return column


def calculate_A(frequency):
    """
    Computes the matrix of shifted bspline functions
    :param frequency:
    :return:
    """
    initialize(frequency)
    column = get_basic_bspline(length)
    A = np.zeros((length_b, configuration.NUMBER_OF_BSPLINE_COMPONENTS))
    # And now copy into A at the correct place
    # Note that since A is populated column by column
    for i in range(configuration.NUMBER_OF_BSPLINE_COMPONENTS):  # 0;i<NUM;i++)
        A[(length * i):(length * i) + 3 * length, i] = column
    return A


def least_square(A, b):
    """
    Returns the result of least square for Ax=b
    :param A:
    :param b:
    :return:
    """
    W, _, _, _ = np.linalg.lstsq(A, b)
    return W


def approximate_bandwidth(center_frequency):
    """
    Given the center frequency returns the bandwidth of the filter
    :param center_frequency: center_frequency of the filter in Hz
    :return: bandwidth of the filter
    """
    # constants for approximation are taken from wiki
    return 24.7 * (4.37 * center_frequency + 1) / 1000


def approximate_gamma_with_spline(frequency, bandwidth=None, normalize=False):
    """
    This method calculates the coefficients of bspline approximation of a gammatone kernel
    :param normalize:
    :param frequency:
    :param bandwidth:
    :return:
    """
    if bandwidth is None:
        bandwidth = approximate_bandwidth(frequency)
    initialize(frequency, bandwidth)
    b = calculate_b(frequency, bandwidth, normalize)
    matrix_a = calculate_A(frequency)
    return least_square(matrix_a, b)


def get_all_kernels(num_kernels, kernel_indexes=None, samp_rate=configuration.sampling_rate, normalize=True):
    """
    This method returns a number of kernels from the filter bank specified by num_kernels
    :param kernel_indexes:
    :param normalize:
    :param samp_rate:
    :param num_kernels: number of kernels returned
    :return:
    """
    if kernel_indexes is None:
        kernel_indexes = get_kernel_indexes(num_kernels)
        all_frequencies = file_utils.read_1D_np_array(filename, ',')
        kernel_frequencies = all_frequencies[kernel_indexes]

    if configuration.verbose:
        print(f' All the kernel frequencies are:', kernel_frequencies)
    kernel_bandwidths = np.array([approximate_bandwidth(f) for f in kernel_frequencies])

    all_kernels = []
    for i in range(num_kernels):
        this_kernel = get_gammatone_kernel(kernel_frequencies[i], kernel_bandwidths[i])
        if normalize:
            this_kernel = signal_utils.normalize_signal(this_kernel, samp_rate)
        all_kernels.append(this_kernel)
    return all_kernels


def get_bspline_kernels(num_kernels, kernel_indexes, load_from_file=False, normalize=True):
    """

    :param num_kernels:
    :param kernel_indexes:
    :param load_from_file:
    :param normalize:
    :return:
    """
    if kernel_indexes is None:
        kernel_indexes = get_kernel_indexes(num_kernels)
    kernel_frequencies = get_kernel_frequencies(kernel_indexes)
    comp_bsplines = []
    comp_bspline_coeffs = np.zeros((num_kernels, configuration.NUMBER_OF_BSPLINE_COMPONENTS))
    comp_bspline_shifts = np.zeros((num_kernels, configuration.NUMBER_OF_BSPLINE_COMPONENTS))
    for i in range(num_kernels):
        initialize(kernel_frequencies[i])
        this_bspline = get_basic_bspline(length)
        comp_bsplines.append(this_bspline)
        for j in range(configuration.NUMBER_OF_BSPLINE_COMPONENTS):
            comp_bspline_shifts[i][j] = j * length
        if load_from_file:
            coeffs = file_utils.read_1D_np_array(COEFFICIENTS_FOLDER_PATH +
                                                 'solution-' + str(kernel_indexes[i]) + '.values', '\n')
            comp_bspline_coeffs[i] = coeffs
        else:
            comp_bspline_coeffs[i] = approximate_gamma_with_spline(kernel_frequencies[i], normalize=normalize)
    return comp_bsplines, comp_bspline_coeffs, comp_bspline_shifts


def get_kernel_frequencies(kernel_indexes=None):
    """
    Returns the list of available frequncies selected by indexes
    :param kernel_indexes:
    :return:
    """
    all_frequencies = file_utils.read_1D_np_array(filename, ',')
    if kernel_indexes is None:
        return all_frequencies
    else:
        return all_frequencies[kernel_indexes]


def get_kernel_indexes(number_of_kernels):
    """
    Returns the kernel indexes from the set of all available kernels
    :param number_of_kernels:
    :return:
    """
    step_interval = total_available_kernels // number_of_kernels
    kernel_indexes = np.arange(0, total_available_kernels, step_interval)
    return kernel_indexes


def write_all_bspline_coeffs(num_kernels, normalize=True, start_index=0, end_index=total_available_kernels - 1):
    """
    :param num_kernels:
    :param normalize:
    :param start_index:
    :type end_index: object
    """

    step_interval = total_available_kernels // num_kernels
    if configuration.verbose:
        print(f'step interval is {step_interval}')
    all_frequencies = file_utils.read_1D_np_array(filename, ',')
    kernel_frequencies = all_frequencies[::step_interval]
    for j, i in enumerate(range(start_index, end_index - 1, step_interval)):
        if configuration.verbose:
            print(f'writing the kernel of center frequency: {kernel_frequencies[j]}')
        comp_bspline_coeffs = approximate_gamma_with_spline(kernel_frequencies[j], normalize=normalize)
        file_utils.write_1D_np_array(COEFFICIENTS_FOLDER_PATH + 'solution-' + str(i) + '.values', comp_bspline_coeffs,
                                     '\n')

# configuration.verbose = True
# load_all_bspline_coeffs(4, normalize=True)
# def get_bspline_for_a_frequency(center_freq):
#
# test_f = 6509.47070101
# test_b = approximate_bandwidth(test_f)
# g_fl = get_gammatone_kernel(test_f, test_b)
# # print(f'the length is {length}')
# # print(approximate_gamma_with_spline(test_f, test_b))
# #plot_utils.plot_function(np.matmul(calculate_A(test_f), approximate_gamma_with_spline(test_f, test_b)))
# # print(f'length of the filter is {length_b}')
# plot_utils.plot_function(signal_utils.normalize_signal(g_fl))
