from distutils.command.config import config

import numpy as np
import signal_utils
import gammatone_calculator
import configuration
import time

all_kernels = []
kernel_inner_products = []
bspline_inner_products = []
kernel_component_positions = []
kernels_component_bsplines = []
kernels_bspline_coefficients = []
component_shifts = []
num_kernels = -1

sample_rate = 0


def init(number_of_kernels, kernel_array=None, sampling_rate=configuration.sampling_rate, normalize=True,
         mode='compressed', load_from_cache=True):
    """
    Initializes a number of kernels by populating the necessary data structures
    :param load_from_cache:
    :param kernel_array: the kernel indexes
    :param normalize:
    :param sampling_rate:
    :param number_of_kernels:
    :param mode:
    :return:
    """
    global all_kernels, kernel_inner_products, kernels_component_bsplines, kernel_component_positions, \
        kernels_bspline_coefficients, component_shifts, bspline_inner_products, num_kernels
    num_comps = configuration.NUMBER_OF_BSPLINE_COMPONENTS
    if mode == 'expanded':
        all_kernels = gammatone_calculator.get_all_kernels(number_of_kernels, kernel_array, normalize=normalize)
        num_kernels = len(all_kernels)
        kernel_inner_products = [[None for i in range(len(all_kernels))] for j in range(len(all_kernels))]
        # convention here is to store the convolution of two kernels- index1, index2 with index1 < index2
        for i in range(len(all_kernels)):
            print(f'norm of the kernel is: {signal_utils.get_signal_norm_square(all_kernels[i])}')
            for j in range(i, len(all_kernels)):
                kernel_inner_products[i][j] = signal_utils.calculate_convolution(all_kernels[i][::-1], all_kernels[j],
                                                                                 sampling_rate)
    elif mode == 'compressed':
        kernels_component_bsplines, kernels_bspline_coefficients, component_shifts = gammatone_calculator. \
            get_bspline_kernels(number_of_kernels, kernel_array, load_from_file=load_from_cache, normalize=normalize)
        bspline_inner_products = [[None for i in range(len(kernels_component_bsplines))] for j in
                                  range(len(kernels_component_bsplines))]
        kernel_component_positions = [[None for i in range(len(kernels_component_bsplines))] for j in
                                      range(len(kernels_component_bsplines))]
        num_kernels = len(kernels_component_bsplines)
        for i in range(len(kernels_component_bsplines)):
            for j in range(i, len(kernels_component_bsplines)):
                f = signal_utils.calculate_convolution(kernels_component_bsplines[i],
                                                       kernels_component_bsplines[j],
                                                       sampling_rate)
                strt = time.process_time()
                positions = np.array([int(len(kernels_component_bsplines[j]) +
                                          component_shifts[j][x]) for x in range(num_comps)], int)
                # print(f'allocation took {time.process_time() - strt}s')
                bspline_inner_products[i][j] = f
                kernel_component_positions[i][j] = positions

                # store the reverse as well, please note that the convolutions are symmetric
                # but to compute the exact inner product the time delta need to be negated
                bspline_inner_products[j][i] = f
                kernel_component_positions[j][i] = positions
    else:
        ValueError(f'unrecognized mode {mode} specified')


def calculate_kernel_ip_from_bspline(index1, index2, time_delta):
    """
    Calculates the inner product of two kernels using the inner product of component bsplines
    :param index1:
    :param index2:
    :param time_delta: by convention kernel index2 is lagging from kernel index1 by time_delta
    :return:
    """
    if configuration.verbose:
        start_time = time.process_time()
    if index1 > index2:
        return calculate_kernel_ip_from_bspline(index2, index1, -time_delta)
    else:
        ip_value = 0
        for c in range(configuration.NUMBER_OF_BSPLINE_COMPONENTS):
            for d in range(configuration.NUMBER_OF_BSPLINE_COMPONENTS):
                comp_shift1 = component_shifts[index1][c]
                comp_shift2 = component_shifts[index2][d]
                delta = time_delta - comp_shift2 + comp_shift1
                position = int(len(kernels_component_bsplines[index2]) - delta)
                if position < 0 or position >= len(bspline_inner_products[index1][index2]):
                    continue
                else:
                    ip_value += kernels_bspline_coefficients[index1][c] * kernels_bspline_coefficients[index2][d] * \
                                bspline_inner_products[index1][index2][position]
    if configuration.verbose:
        end_time = time.process_time()
        print(f'time taken to get the kernel kernel ip: {end_time - start_time}')
    return ip_value


def calculate_row_of_kernel_ips(index1, time_deltas, spike_indexes):
    """
        Calculates the inner product of two kernels using the inner product of component bsplines
        :param index1:
        :param index2:
        :param time_delta: by convention kernel index2 is lagging from kernel index1 by time_delta
        :return:
        """
    num_comps = configuration.NUMBER_OF_BSPLINE_COMPONENTS
    ############################### carefully handle the case where index1 < index2##################################
    initial_positions = np.array([[int(len(kernels_component_bsplines[spike_indexes[i]]) - (time_deltas[i] -
                component_shifts[spike_indexes[i]][j])) for j in range(num_comps)] for i in range(len(time_deltas))])
    all_positions = np.array([[initial_positions[n] - int(component_shifts[index1][1] * i)
                               for i in range(num_comps)] for n in range(len(time_deltas))])
    all_positions[(all_positions <= 0) | (all_positions >= len(bspline_inner_products[index1][index2]))] = 0
    all_positions = np.ravel(all_positions)
    all_ips = np.zeros_like(all_positions, dtype=float)
    # if configuration.verbose:
    #     start_time = time.process_time()
    # num_comps = configuration.NUMBER_OF_BSPLINE_COMPONENTS
    # # if index1 > index2:
    # #     return calculate_kernel_ip_from_bspline(index2, index1, -time_delta)
    # # else:
    # all_positions = np.array(
    #     [[(kernel_component_positions[index1][spike_indexes[n]] - int(component_shifts[index1][i]
    #                                                                   if index1 < spike_indexes[n] else
    #                                                                   component_shifts[spike_indexes[n]][i]) -
    #        int(time_deltas[n] if index1 < spike_indexes[n] else - time_deltas[n])) for i in
    #       range(num_comps)] for n in range(len(time_deltas))])
    # all_positions[(all_positions <= 0) | (all_positions >= len(bspline_inner_products[index1][index2]))] = 0
    # all_positions = np.ravel(all_positions)
    # all_ips = np.zeros_like(all_positions, dtype=float)
    #
    # all_ips[all_positions > 0] = bspline_inner_products[index1][index2][all_positions[all_positions > 0]]
    # all_ips = all_ips.reshape(configuration.NUMBER_OF_BSPLINE_COMPONENTS, -1)
    # print(f'{all_ips}')
    # ip_vector = np.dot(kernels_bspline_coefficients[index1],
    #                    np.dot(all_ips, kernels_bspline_coefficients[index2].T).T)
    # if configuration.verbose:
    #     end_time = time.process_time()
    #     print(f'time taken to get the vectorized kernel kernel ip: {end_time - start_time}')
    # return ip_vector


def calculate_kernel_ip_from_bspline_efficient(index1, index2, time_delta):
    """
    Calculates the inner product of two kernels using the inner product of component bsplines
    :param index1:
    :param index2:
    :param time_delta: by convention kernel index2 is lagging from kernel index1 by time_delta
    :return:
    """
    if configuration.verbose:
        start_time = time.process_time()
    num_comps = configuration.NUMBER_OF_BSPLINE_COMPONENTS
    if index1 > index2:
        return calculate_kernel_ip_from_bspline(index2, index1, -time_delta)
    else:
        # all_positions = np.array(
        #     [(kernel_component_positions[index1][index2] - int(component_shifts[index1][i]) - int(time_delta)) for i in
        #      range(num_comps)])

        initial_pos = np.array([int(len(kernels_component_bsplines[index2]) - (time_delta -
                                                                               component_shifts[index2][j])) for j in
                                range(num_comps)])
        all_positions = np.array([initial_pos - int(component_shifts[index1][1]*i) for i in range(num_comps)])
        all_positions[(all_positions <= 0) | (all_positions >= len(bspline_inner_products[index1][index2]))] = 0
        all_positions = np.ravel(all_positions)
        all_ips = np.zeros_like(all_positions, dtype=float)

        all_ips[all_positions > 0] = bspline_inner_products[index1][index2][all_positions[all_positions > 0]]
        all_ips = all_ips.reshape(configuration.NUMBER_OF_BSPLINE_COMPONENTS, -1)
        print(f'{all_ips}')
        ip_vector = np.dot(kernels_bspline_coefficients[index1],
                           np.dot(all_ips, kernels_bspline_coefficients[index2].T).T)
    if configuration.verbose:
        end_time = time.process_time()
        print(f'time taken to get the vectorized kernel kernel ip: {end_time - start_time}')
    return ip_vector


def get_position(index1, index2, x, time_delta):
    """
    This function returns the index location where the inner prod value
    needs to be checked
    :param index1:
    :param index2:
    :param x:
    :param time_delta:
    :return:
    """
    c = int(x / configuration.NUMBER_OF_BSPLINE_COMPONENTS)
    d = int(x % configuration.NUMBER_OF_BSPLINE_COMPONENTS)
    delta = time_delta - component_shifts[index2][d] + component_shifts[index1][c]
    pos = int(len(kernels_component_bsplines[index2]) - delta) if 0 < int(
        len(kernels_component_bsplines[index2]) - delta) < len(bspline_inner_products[index1][index2]) else 0
    return pos


def get_initial_position(index1, index2, x):
    """
    This function returns the index location where the inner prod value
    needs to be checked
    :param index1:
    :param index2:
    :param x:
    :param time_delta:
    :return:
    """
    c = int(x / configuration.NUMBER_OF_BSPLINE_COMPONENTS)
    d = int(x % configuration.NUMBER_OF_BSPLINE_COMPONENTS)
    delta = - component_shifts[index2][d] + component_shifts[index1][c]
    return int(len(kernels_component_bsplines[index2]) - delta)


def fetch_kernel_ip_directly(index1, index2, time_delta):
    """
    Returns the inner product of two kernels directly based on the stored convolution values
    :param index1:
    :param index2:
    :param time_delta: by convention kernel index2 is lagging from kernel index1 by time_delta
    :return:
    """
    if index1 > index2:
        return fetch_kernel_ip_directly(index2, index1, -time_delta)
    else:
        position = int(len(all_kernels[index1]) - 1 + time_delta)
        if position < 0 or position >= len(kernel_inner_products[index1][index2]):
            return 0
        else:
            return kernel_inner_products[index1][index2][position]


def get_kernel_inner_prod_signal(index1, index2, mode='compressed', start=None, end=None):
    if start is None:
        start = -len(all_kernels[index1])
    if end is None:
        end = len(all_kernels[index2])
    signal = np.zeros(end - start + 1)
    for i, delta in enumerate(range(start, end + 1)):
        if mode == 'compressed':
            signal[i] = get_kernel_kernel_inner_prod(index1, index2, delta, 'compressed')
        elif mode == 'expanded':
            signal[i] = fetch_kernel_ip_directly(index1, index2, delta)
    return signal


def get_kernel_kernel_inner_prod(index1, index2, time_delta, mode="expanded"):
    """
    Returns the inner product of two kernels separated by a time_delta. By convention kernel index2 is lagging from
    kernel index1 by amount time_delta
    :param index1:
    :param index2:
    :param time_delta:
    :param mode:
    :return:
    """
    if mode == 'compressed':
        return calculate_kernel_ip_from_bspline(index1, index2, time_delta)
    elif mode == 'expanded':
        return fetch_kernel_ip_directly(index1, index2, time_delta)
    else:
        raise ValueError(f'incorrect mode: {mode} has been specified')
