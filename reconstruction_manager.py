import numpy as np
import torch

import kernel_manager
import signal_utils
import spike_generator
import common_utils
import time
import file_utils
import configuration
import plot_utils
import gammatone_calculator
import tensorflow as tf
import iterative_spike_generator


def calculate_signal_kernel_bspline_convs(signal, kernels_component_bsplines):
    all_convolutions = []
    print(f'length of all kernels {len(kernels_component_bsplines)}')
    for i in range(len(kernels_component_bsplines)):
        all_convolutions.append(signal_utils.calculate_convolution(kernels_component_bsplines[i], signal))
        print(f'size of all convs {len(all_convolutions)}')
    return all_convolutions


def init_signal(signal, mode=configuration.mode):
    """
    Initialize the necessary global data structures for the given signal
    :param mode:
    :return:
    :param signal:
    """
    signal_norm_square = signal_utils.get_signal_norm_square(signal)
    signal_kernel_convolutions = calculate_signal_kernel_convs(signal, mode)
    return signal_norm_square, signal_kernel_convolutions  # , signal_kernel_bspline_convolutions


def calculate_signal_kernel_convs(signal, mode=configuration.mode):
    """
    This function calculates the convolution of a signal with the given kernels
    :param mode:
    :param signal:
    :return:
    """
    all_convolutions = []
    for i in range(kernel_manager.num_kernels):
        if mode == 'expanded':
            all_convolutions.append(signal_utils.calculate_convolution(kernel_manager.all_kernels[i], signal))
        elif mode == 'compressed':
            all_convolutions.append(signal_utils.add_shifted_comp_signals(
                signal_utils.calculate_convolution(kernel_manager.kernels_component_bsplines[i], signal),
                kernel_manager.component_shifts[i], kernel_manager.kernels_bspline_coefficients[i]))
        print(f'size of all convs {len(all_convolutions)}')
    return all_convolutions


def calculate_reconstruction(spike_times, spike_indexes, threshold_crossing_values):
    """
    Once spike times are computed this method computes the reconstruction coefficients
    """
    p_matrix = calculate_p_matrix(spike_times, spike_indexes)

    # p_inv = common_utils.solve_for_inverse_by_torch(p_matrix)
    # reconstruction_coefficients = np.dot(p_inv, threshold_crossing_values)
    reconstruction_coefficients = common_utils.solve_for_coefficients(p_matrix, threshold_crossing_values).numpy()
    return reconstruction_coefficients


def calculate_p_matrix(spike_times, spike_indexes, mode=configuration.mode):
    """
    This method populates the entries of the p_matrix
    :param mode:
    :param spike_times:
    :type spike_indexes:
    """
    if configuration.verbose:
        start_time = time.process_time()
    n = len(spike_times)
    p_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            try:
                p_matrix[i][j] = kernel_manager.get_kernel_kernel_inner_prod(
                    spike_indexes[i], spike_indexes[j], spike_times[j] - spike_times[i], mode=mode)
            except Exception as e:
                raise ValueError(f' exception occurred for P matrix entry of {i}th and {j}th spikes'
                                 f'at times{spike_times[i]} and {spike_times[j]} with exception: {e}')
    if configuration.verbose:
        end_time = time.process_time()
        print(f'time taken to compute p_matrix: {end_time - start_time}')
    return p_matrix


def calculate_spike_times_and_reconstruct(signal_kernel_convs, reconstruction=True,
                                          ahp_period=configuration.ahp_period, selected_kernel_indexes=None):
    """
    This method calculates the spike times and calculates the reconstruction coefficients
    :param selected_kernel_indexes:
    :param ahp_period:
    :param signal_kernel_convs:
    :param reconstruction:
    :return:
    """
    spike_times, spike_indexes, threshold_values = spike_generator. \
        calculate_spike_times(signal_kernel_convs, ahp_period=ahp_period,
                              selected_kernel_indexes=selected_kernel_indexes)
    if len(spike_times) == 0:
        return spike_times, spike_indexes, None, None
    recons_coeffs = None
    if reconstruction:
        recons_coeffs = calculate_reconstruction(spike_times, spike_indexes, threshold_values)
    return spike_times, spike_indexes, threshold_values, recons_coeffs


def get_reconstructed_signal(signal_length, spike_times, spike_indexes, reconstruction_coefficients):
    """
    Returns the reconstruction signal of same length as the original signal
    :param signal_length:
    :param spike_times:
    :param spike_indexes:
    :param reconstruction_coefficients:
    :return: returns the reconstructed signal
    """
    if configuration.verbose:
        start_time = time.process_time()
    reconstructed_signal = np.zeros(signal_length)
    last_signal_index = signal_length - 1
    for i in range(len(spike_times)):
        # print(f'adding the component number: {i}')
        this_component = reconstruction_coefficients[i] * kernel_manager.all_kernels[spike_indexes[i]]
        ti = int(spike_times[i])
        reconstructed_signal[int(np.maximum(ti - len(this_component) + 1, 0)):int(np.minimum(ti, last_signal_index))] = \
            reconstructed_signal[
            int(np.maximum(ti - len(this_component) + 1, 0)):int(np.minimum(ti, last_signal_index))] \
            + this_component[int(np.minimum(len(this_component) - 1, ti)):int(np.maximum(0, ti - last_signal_index)):-1]
    if configuration.verbose:
        print(f'reconstruction took: {time.process_time() - start_time}s')
    return reconstructed_signal


def calculate_reconstruction_error_rate_fast(reconstruction_coefficients, threshold_crossing_values,
                                             signal_norm_square):
    """
    This method computes the norm square of the error signal divided by the squared norm of the original signal
    :return:
    :param threshold_crossing_values:
    :param reconstruction_coefficients:
    :type signal_norm_square: object
    :return:
    """
    return (signal_norm_square - np.dot(np.ravel(reconstruction_coefficients),
                                        np.ravel(threshold_crossing_values))) / signal_norm_square


def drive_single_signal_reconstruction(signal, init_kernel=True, number_of_kernels=-1, kernel_frequencies=None,
                                       need_error_rate_fast=True, need_error_rate_accurate=False,
                                       need_reconstructed_signal=False, computation_mode=configuration.mode,
                                       ahp_period=configuration.ahp_period, selected_kernel_indexes=None):
    """

    :param selected_kernel_indexes:
    :param ahp_period:
    :param computation_mode:
    :param signal:
    :param init_kernel:
    :param number_of_kernels:
    :param kernel_frequencies:
    :param need_error_rate_fast:
    :param need_error_rate_accurate:
    :param need_reconstructed_signal:
    :return:
    """
    if init_kernel:
        kernel_manager.init(number_of_kernels, kernel_frequencies)
    signal_norm_square, signal_kernel_convolutions = init_signal(signal, computation_mode)
    sp_times, sp_indexes, thrs_values, recons_coeffs = calculate_spike_times_and_reconstruct(
        signal_kernel_convolutions, need_error_rate_fast, ahp_period=ahp_period,
        selected_kernel_indexes=selected_kernel_indexes)
    recons = None
    error_rate_fast = None
    if need_error_rate_accurate:
        pass
    if need_error_rate_fast:
        error_rate_fast = calculate_reconstruction_error_rate_fast(recons_coeffs, thrs_values, signal_norm_square)
    if need_reconstructed_signal:
        recons = get_reconstructed_signal(len(signal), sp_times, sp_indexes, recons_coeffs)
    return sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons


def drive_single_signal_reconstruction_iteratively(signal, init_kernel=True, number_of_kernels=-1,
                                                   kernel_frequencies=None,
                                                   need_error_rate_fast=True, need_error_rate_accurate=False,
                                                   need_reconstructed_signal=False, computation_mode=configuration.mode,
                                                   window_mode=False, window_size=-1, norm_threshold=0.01,
                                                   z_threshold=0.5, recompute_recons_coeff=True, show_z_vals=False,
                                                   signal_norm_sq=None, selected_kernel_indexes=None):
    """
    This method uses iterative technique to generate spikes and reconstruct a signal
    :param selected_kernel_indexes:
    :param signal_norm_sq:
    :param z_threshold:
    :param show_z_vals:
    :param recompute_recons_coeff:
    :param norm_threshold: threshold for inner product of the new spike in the current span
    :param computation_mode:
    :param window_size:
    :param window_mode:
    :param signal:
    :param init_kernel:
    :param number_of_kernels:
    :param kernel_frequencies:
    :param need_error_rate_fast:
    :param need_error_rate_accurate:
    :param need_reconstructed_signal:
    :return:
    """
    if init_kernel:
        kernel_manager.init(number_of_kernels, kernel_frequencies)
    signal_norm_square, signal_kernel_convolutions = init_signal(signal, computation_mode)
    sp_times, sp_indexes, thrs_values, recons_coeffs, z_scores, kernel_projections, gamma_vals = \
        iterative_spike_generator. \
        spike_and_reconstruct_iteratively(signal_kernel_convolutions, window_mode=window_mode, window_size=window_size,
                                          norm_threshold_for_new_spike=norm_threshold, z_thresholds=z_threshold,
                                          show_z_scores=show_z_vals, signal_norm_square=signal_norm_sq,
                                          selected_kernel_indexes=selected_kernel_indexes)
    if recompute_recons_coeff:
        recons_coeffs = calculate_reconstruction(sp_times, sp_indexes, thrs_values)
    recons = None
    error_rate_fast = None
    if need_error_rate_accurate:
        pass
    if need_error_rate_fast:
        error_rate_fast = calculate_reconstruction_error_rate_fast(recons_coeffs, thrs_values, signal_norm_square)
    if need_reconstructed_signal:
        recons = get_reconstructed_signal(len(signal), sp_times, sp_indexes, recons_coeffs)
    return sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons, \
           signal_kernel_convolutions, z_scores, kernel_projections, gamma_vals

# # configuration.upsample_factor = 10
# # configuration.ahp_period = 100.0 * configuration.upsample_factor
# # configuration.ahp_high_value = 1000.0 * configuration.upsample_factor
# # configuration.spiking_threshold = 5.0e-6
# configuration.verbose = True

# ind1 = 2
# ind2 = 3
# compressed = False
#
# import random
#
# itr = 20
# index1 = random.randint(0, number_of_kernel - 1)
# index2 = random.randint(0, number_of_kernel - 1)
# rand_time_delta = 0  # random.randint(0, len(kernel_manager.all_kernels[index1]) + len(kernel_manager.all_kernels[index2]) - 1)
# ip1 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta, mode='compressed')
# print(f'{ip1}: This is the compressed inner products')
# ip2 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta)
# print(f'{ip2}: This is the expanded inner products')
# print(f'{0} iteration completed for time delta: {rand_time_delta}')
# st = -2000
# end = 3000
# signal_ext = kernel_manager.get_kernel_inner_prod_signal(ind1, ind2, mode='expanded', start=st, end=end)
# # signal_comp = kernel_manager.get_kernel_inner_prod_signal(ind1, ind2, mode='compressed', start=st, end=end)
# plot_utils.plot_function(signal_ext, title=f'expand signal inner product for kernels {ind1} and {ind2}')
# plot_utils.plot_function(signal_comp, title=f'compression signal inner product for kernels {ind1} and {ind2}')


# for j, i in enumerate(range(st, end)):
#     index1 = ind1  # random.randint(0, number_of_kernel - 1)
#     index2 = ind2  # random.randint(0, number_of_kernel - 1)
#     rand_time_delta = i
#     ip1 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta, mode='compressed')
#     ip2 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta)
#     print(f'{j},{ip1}')
#     # print(f'The compressed and the expanded inner products are: ({ip1},{ip2})')
#     # print(f'{i} iteration completed for time delta: {rand_time_delta}')
# for i in range(itr):
#     index1 = random.randint(0, number_of_kernel - 1)
#     index2 = random.randint(0, number_of_kernel - 1)
#     rand_time_delta = random.randint(0, len(kernel_manager.all_kernels[index1]) + len(
#         kernel_manager.all_kernels[index2]) - 1)
#     ip1 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta, mode='compressed')
#     print(f'{ip1}: This is the compressed inner products')
#     ip2 = kernel_manager.get_kernel_kernel_inner_prod(index1, index2, rand_time_delta)
#     print(f'{ip2}: This is the expanded inner products')
#     print(f'{i} iteration completed for time delta: {rand_time_delta}')
# assert ip1 == ip2
# for i in range(number_of_kernel):
#     added_bspline = signal_utils.add_shifted_comp_signals(kernel_manager.kernels_component_bsplines[i],
#                                                           kernel_manager.component_shifts[i],
#                                                           kernel_manager.kernels_bspline_coefficients[i])
#     convolution_test = signal_utils.calculate_convolution(test_signal, added_bspline)
#     component_wise_convolution = signal_utils.add_shifted_comp_signals(bspline_convs[i],
#                                                                        kernel_manager.component_shifts[i],
#                                                                        kernel_manager.kernels_bspline_coefficients[i])
#     plot_utils.plot_function(component_wise_convolution, title="added convolution signal")
#     convolution_on_full_signal = signal_utils.calculate_convolution(added_bspline, test_signal)
#     plot_utils.plot_function(convolution_on_full_signal, title="convolution on full signal")
