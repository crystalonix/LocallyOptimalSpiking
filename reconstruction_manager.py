import time

import numpy as np

import common_utils
import configuration
import iterative_spike_generator
import kernel_manager
import signal_utils
import spike_generator
import plot_utils
import math
import concurrent.futures
import itertools


def calculate_signal_kernel_bspline_convs(signal, kernels_component_bsplines):
    all_convolutions = []
    print(f'length of all kernels {len(kernels_component_bsplines)}')
    for i in range(len(kernels_component_bsplines)):
        all_convolutions.append(signal_utils.calculate_convolution(kernels_component_bsplines[i], signal))
        print(f'size of all convs {len(all_convolutions)}')
    return all_convolutions


def init_signal(signal, mode=configuration.mode, need_norm=True, select_kernel_indexes=None):
    """
    Initialize the necessary global data structures for the given signal
    :param select_kernel_indexes:
    :param need_norm:
    :param mode:
    :return:
    :param signal:
    """
    signal_norm_square = -1
    if need_norm:
        signal_norm_square = signal_utils.get_signal_norm_square(signal)
    if configuration.parallel_convolution:
        signal_kernel_convolutions = calculate_signal_kernel_convs_parallel(signal, select_kernel_indexes)
    else:
        signal_kernel_convolutions = calculate_signal_kernel_convs(signal, mode, select_kernel_indexes)
    return signal_norm_square, signal_kernel_convolutions  # , signal_kernel_bspline_convolutions


def calculate_signal_kernel_convs(signal, mode=configuration.mode, select_kernel_indexes=None):
    """
    This function calculates the convolution of a signal with the given kernels
    :param select_kernel_indexes:
    :param mode:
    :param signal:
    :return:
    """
    all_convolutions = []
    for i in range(kernel_manager.num_kernels):
        if select_kernel_indexes is not None and i not in select_kernel_indexes:
            all_convolutions.append([])
            continue
        if mode == 'expanded':
            all_convolutions.append(signal_utils.calculate_convolution(kernel_manager.all_kernels[i], signal))
        elif mode == 'compressed':
            all_convolutions.append(signal_utils.add_shifted_comp_signals(
                signal_utils.calculate_convolution(kernel_manager.kernels_component_bsplines[i], signal),
                kernel_manager.component_shifts[i], kernel_manager.kernels_bspline_coefficients[i]))
        # print(f'size of all convs {len(all_convolutions)}')
    return all_convolutions


def calculate_signal_kernel_convs_parallel(signal, select_kernel_indexes=None):
    """
    This function calculates the convolution of a signal with the given kernels in parallel mode
    :param select_kernel_indexes:
    :param signal:
    :return:
    """
    all_convolutions = [None for i in range(kernel_manager.num_kernels)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=configuration.number_of_threads) as executor:
        for i, result in enumerate(executor.map(get_conv,
                                                [i for i in range(kernel_manager.num_kernels)],
                                                itertools.repeat(signal), itertools.repeat(select_kernel_indexes))):
            all_convolutions[i] = result
    return all_convolutions


def get_conv(kernel_index, signal, select_kernel_indexes):
    """
    Used in process executor to return the signal kernel convolutions
    :param select_kernel_indexes:
    :param kernel_index:
    :type signal: object
    """
    if select_kernel_indexes is not None and kernel_index not in select_kernel_indexes:
        return []
    else:
        return signal_utils.calculate_convolution(kernel_manager.all_kernels[kernel_index], signal)


def calculate_reconstruction(spike_times, spike_indexes, threshold_crossing_values):
    """
    Once spike times are computed this method computes the reconstruction coefficients
    """
    p_matrix = calculate_p_matrix(spike_times, spike_indexes)

    # p_inv = common_utils.solve_for_inverse_by_torch(p_matrix)
    # reconstruction_coefficients = np.dot(p_inv, threshold_crossing_values)
    reconstruction_coefficients = common_utils.solve_for_coefficients(p_matrix, threshold_crossing_values).numpy()
    return reconstruction_coefficients


def calculate_reconstruction_in_window_mode(spike_times, spike_indexes, threshold_crossing_values,
                                            winddow_size=configuration.window_size):
    """
    Once spike times are computed this method computes the reconstruction coefficients
    """
    step_len = 1000
    recons_coeffs = np.zeros(len(spike_times))
    p_matrix = calculate_p_matrix(spike_times[: winddow_size], spike_indexes[: winddow_size])
    recons_coeffs[:min(winddow_size, len(spike_times))] = np.squeeze(
        common_utils.solve_for_coefficients(p_matrix,
                                            threshold_crossing_values[:min(winddow_size, len(spike_times))]).numpy())
    for i in range(winddow_size, len(spike_times)):
        # eta_vals = calculate_eta(spike_times[:i], spike_indexes[:i], spike_times[i], spike_indexes[i], True,
        #                          winddow_size - 1)
        all_eta_vals = calculate_eta(spike_times[:i], spike_indexes[:i], spike_times[i], spike_indexes[i], False)
        eta_vals = all_eta_vals[-winddow_size + 1:]
        p_matrix = update_windowed_p_matrix(p_matrix, eta_vals, winddow_size)
        t_values = np.zeros(winddow_size)
        t_values[-1] = \
            threshold_crossing_values[i] - \
            np.dot(recons_coeffs[:i], all_eta_vals)
        # np.dot(recons_coeffs[max(i - winddow_size+1, 0):i], eta_vals)
        coeffs = common_utils.solve_for_coefficients(p_matrix, t_values).numpy()
        recons_coeffs[max(i + 1 - winddow_size, 0):i + 1] = recons_coeffs[max(i + 1 - winddow_size, 0):i + 1] \
                                                            + np.squeeze(coeffs)
        if configuration.debug and i % step_len == 0:
            recons = get_reconstructed_signal(spike_times[i], spike_times[:i + 1], spike_indexes[:i + 1], recons_coeffs)
            plot_utils.plot_function(recons)
            # absolute_error_rate = signal_utils.calculate_absolute_error_rate(signal, recons)
            # print(f'absolute error rate: {absolute_error_rate}')
        # TODO: add error calculation here
    return recons_coeffs


def calculate_reconstruction_in_batch_window_mode(spike_times, spike_indexes, threshold_crossing_values,
                                                  window_size=configuration.window_size,
                                                  batch_size=configuration.window_spike_batch_size):
    """
    Once spike times are computed this method computes the reconstruction coefficients
    """
    step_len = 1000
    recons_coeffs = np.zeros(len(spike_times))
    p_matrix = calculate_p_matrix(spike_times[: batch_size], spike_indexes[: batch_size])
    recons_coeffs[:min(batch_size, len(spike_times))] = np.squeeze(
        common_utils.solve_for_coefficients(p_matrix,
                                            threshold_crossing_values[:min(batch_size, len(spike_times))]).numpy())
    for b in range(1, math.ceil(len(spike_times) / batch_size)):
        start_index = max(batch_size * b - window_size, 0)
        window_offset = batch_size * b - start_index
        end_index = min(batch_size * (b + 1), len(spike_times))
        spikes_to_consider = spike_times[start_index:end_index]
        indexes_to_consider = spike_indexes[start_index:end_index]
        if configuration.compute_time:
            t_1 = time.time()
        if configuration.parallel_convolution:
            p_matrix = update_p_matrix_in_windowed_batch_mode_parallel(p_matrix,
                                                                       spikes_to_consider, indexes_to_consider,
                                                                       window_offset)
        else:
            p_matrix = update_p_matrix_in_windowed_batch_mode(p_matrix,
                                                              spikes_to_consider, indexes_to_consider, window_offset)
        if configuration.debug:
            print(f'time for p_matrix update: {time.time() - t_1}')
            t_1 = time.time()

        windowed_eta_vals = p_matrix[:window_offset, window_offset:]
        t_vector = np.array(threshold_crossing_values[start_index + window_offset:end_index]) - \
            np.matmul(windowed_eta_vals.T, recons_coeffs[start_index: start_index + window_offset])

        t_vector_all = np.zeros(len(spikes_to_consider))
        t_vector_all[window_offset:] = t_vector
        if configuration.debug:
            print(f'time for t_vector update: {time.time() - t_1}')
            t_1 = time.time()
        coeffs = common_utils.solve_for_coefficients(p_matrix, t_vector_all).numpy()
        recons_coeffs[start_index:end_index] = \
            recons_coeffs[start_index:end_index] + np.squeeze(coeffs)
        if configuration.debug:
            print(f'time for recons update: {time.time() - t_1}')
            t_1 = time.time()
    return recons_coeffs


def update_p_matrix_in_windowed_batch_mode(p_matrix, spikes_to_consider,
                                           indexes_to_consider, window_offset):
    """
    This method updates an existing p_matrix to a new one based on a new set of spikes
    :param window_offset:
    :param spikes_to_consider:
    :param indexes_to_consider:
    :type p_matrix
    """
    n_dim = len(spikes_to_consider)
    updated_p_mat = np.zeros((n_dim, n_dim))
    updated_p_mat[:window_offset, :window_offset] = p_matrix[-window_offset:, -window_offset:]
    all_eta_vals = []
    for i in range(window_offset, n_dim):
        eta_this = calculate_eta(spikes_to_consider[:window_offset], indexes_to_consider[:window_offset],
                                 spikes_to_consider[i], indexes_to_consider[i])
        if len(all_eta_vals) == 0:
            all_eta_vals.append(eta_this)
            all_eta_vals = np.reshape(all_eta_vals, (-1, 1))
        else:
            all_eta_vals = np.hstack((all_eta_vals, np.array([eta_this]).T))
    # include eta values in p_matrix update
    updated_p_mat[:window_offset, window_offset:] = all_eta_vals
    # calculate the new block of the p_matrix
    batch_p = calculate_p_matrix(spikes_to_consider[window_offset:], indexes_to_consider[window_offset:])
    updated_p_mat[window_offset:, window_offset:] = batch_p
    return updated_p_mat


def update_p_matrix_in_windowed_batch_mode_parallel(p_matrix, spikes_to_consider,
                                                    indexes_to_consider, window_offset):
    """
    This method updates an existing p_matrix to a new one based on a new set of spikes
    :param window_offset:
    :param spikes_to_consider:
    :param indexes_to_consider:
    :type p_matrix
    """
    n_dim = len(spikes_to_consider)
    updated_p_mat = np.zeros((n_dim, n_dim))
    updated_p_mat[:window_offset, :window_offset] = p_matrix[-window_offset:, -window_offset:]
    all_eta_vals = []
    results = [None for i in range(n_dim - window_offset)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=configuration.number_of_threads) as executor:
        for k, result in enumerate(executor.map(calculate_eta, itertools.repeat(spikes_to_consider[:window_offset]),
                                                itertools.repeat(indexes_to_consider[:window_offset]),
                                                spikes_to_consider[window_offset:n_dim],
                                                indexes_to_consider[window_offset:n_dim])):
            results[k] = result
    all_eta_vals = np.array(results).T
    # include eta values in p_matrix update
    updated_p_mat[:window_offset, window_offset:] = all_eta_vals
    # calculate the new block of the p_matrix
    if configuration.parallel_convolution:
        batch_p = calculate_p_matrix_parallel(spikes_to_consider[window_offset:], indexes_to_consider[window_offset:])
    else:
        batch_p = calculate_p_matrix(spikes_to_consider[window_offset:], indexes_to_consider[window_offset:])
    updated_p_mat[window_offset:, window_offset:] = batch_p
    return updated_p_mat


# TODO: move this to a separate file
def calculate_eta(spike_times, spike_indexes, current_time, current_index, window_mode=False, window_size=-1):
    """
    This method returns the array of inner product values of
    the current kernel with rest of spike generating kernels
    :param window_size: the maximum number of recent spikes to be considered
    :param window_mode: if the window mode is "True" only a limited set of spikes will be considered for
    calculating the inner products
    :param spike_times: timing of all spikes
    :param spike_indexes: indexes of the spiking kernels
    :param current_time: present time
    :param current_index: index of kernel in consideration
    :return: vector of inner products
    """

    if len(spike_times) == 0:
        return []
    n = min(window_size, len(spike_indexes)) if window_mode else len(spike_indexes)
    eta_values = np.zeros(n)
    for i in range(n):
        this_index = len(spike_indexes) - n + i
        eta_values[i] = kernel_manager. \
            get_kernel_kernel_inner_prod(spike_indexes[this_index], current_index,
                                         current_time - spike_times[this_index])
    return eta_values


# TODO: move this to a common place
def update_windowed_p_matrix(p_matrix, eta_values, window_size=configuration.window_size):
    """
    This method updates
    :param window_size:
    :rtype: return the updated p_matrix and its inverse
    :param p_matrix:
    :param eta_values:
    """
    if len(p_matrix) == 0:
        p_new = np.array([[1.0]])
        return p_new
    # if len(eta_values) != p_matrix.shape[0]:
    #     raise ValueError('P matrix and eta value sizes did not match')
    # assert p_matrix.shape[0] == len(eta_values)
    # TODO: check if this deep copy is needed
    eta_values_new = eta_values.copy()
    p_new = p_matrix[-min(window_size - 1, len(eta_values)):, -min(window_size - 1, len(eta_values)):]
    p_new = np.hstack((p_new, eta_values_new.reshape(-1, 1)))
    # the last element in the matrix is always 1.0
    p_new = np.vstack((p_new, np.append(eta_values_new, 1.0)))
    return p_new


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


def calculate_p_matrix_parallel(spike_times, spike_indexes, mode=configuration.mode):
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=configuration.number_of_threads) as executor:
        for i, result in enumerate(executor.map(get_p_row, [i for i in range(n)], itertools.repeat(spike_times),
                                                itertools.repeat(spike_indexes), itertools.repeat(mode))):
            p_matrix[i] = result
    if configuration.verbose:
        end_time = time.process_time()
        print(f'time taken to compute p_matrix: {end_time - start_time}')
    return p_matrix


def get_p_row(spike_index, spike_times, spike_indexes, mode=configuration.mode):
    """
    This method returns a row of the p-matrix
    :param spike_index:
    :param spike_times:
    :param spike_indexes:
    :param mode:
    :return:
    """
    this_row = np.zeros(len(spike_times))
    for j in range(len(spike_times)):
        try:
            this_row[j] = kernel_manager.get_kernel_kernel_inner_prod(
                spike_indexes[spike_index], spike_indexes[j], spike_times[j] - spike_times[spike_index], mode=mode)
        except Exception as e:
            raise ValueError(f' exception occurred for P matrix entry of {spike_index}th and {j}th spikes'
                             f'at times{spike_times[spike_index]} and {spike_times[j]} with exception: {e}')
    return this_row


def calculate_spike_times_and_reconstruct(signal_kernel_convs, reconstruction=True,
                                          ahp_period=configuration.ahp_period, ahp_high=configuration.ahp_high_value
                                          , selected_kernel_indexes=None,
                                          spiking_threshold=configuration.spiking_threshold,
                                          max_spike_count=configuration.max_spike_count, window_mode=True,
                                          window_size=configuration.window_size):
    """
    This method calculates the spike times and calculates the reconstruction coefficients
    :param window_size:
    :param window_mode:
    :param max_spike_count:
    :param ahp_high:
    :param spiking_threshold:
    :param selected_kernel_indexes:
    :param ahp_period:
    :param signal_kernel_convs:
    :param reconstruction:
    :return:
    """
    spike_times, spike_indexes, threshold_values = spike_generator. \
        calculate_spike_times(signal_kernel_convs, ahp_period=ahp_period, ahp_high=ahp_high,
                              selected_kernel_indexes=selected_kernel_indexes, threshold=spiking_threshold)
    if len(spike_times) == 0 or len(spike_times) > max_spike_count:
        return spike_times, spike_indexes, None, None
    recons_coeffs = None
    if reconstruction:
        if window_mode:
            recons_coeffs = calculate_reconstruction_in_window_mode(spike_times, spike_indexes,
                                                                    threshold_values, winddow_size=window_size)
        else:
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
                                       ahp_period=configuration.ahp_period, ahp_high=configuration.ahp_high_value,
                                       selected_kernel_indexes=None,
                                       spiking_threshold=configuration.spiking_threshold, signal_norm_square=None,
                                       signal_kernel_convolutions=None, max_spike_count=configuration.max_spike_count,
                                       window_mode=True, window_size=configuration.window_size):
    """

    :param window_size:
    :param window_mode:
    :param max_spike_count:
    :param ahp_high:
    :param signal_norm_square:
    :param signal_kernel_convolutions:
    :param spiking_threshold:
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
    recons = None
    if init_kernel:
        kernel_manager.init(number_of_kernels, kernel_frequencies)
    if signal_kernel_convolutions is None:
        signal_norm_square, signal_kernel_convolutions = init_signal(signal, computation_mode)
    sp_times, sp_indexes, thrs_values, recons_coeffs = calculate_spike_times_and_reconstruct(
        signal_kernel_convolutions, need_error_rate_fast, ahp_period=ahp_period, ahp_high=ahp_high,
        selected_kernel_indexes=selected_kernel_indexes, spiking_threshold=spiking_threshold,
        max_spike_count=max_spike_count, window_mode=window_mode, window_size=window_size)

    if need_error_rate_fast and recons_coeffs is not None:
        error_rate_fast = calculate_reconstruction_error_rate_fast(recons_coeffs, thrs_values, signal_norm_square)
    else:
        error_rate_fast = -1
    absolute_error_rate = -1
    if (need_reconstructed_signal or need_error_rate_accurate) and recons_coeffs is not None:
        recons = get_reconstructed_signal(len(signal), sp_times, sp_indexes, recons_coeffs)
        absolute_error_rate = signal_utils.calculate_absolute_error_rate(signal, recons)
        print(f'absolute error rate: {absolute_error_rate}')
    return sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons, absolute_error_rate


def drive_piecewise_signal_reconstruction(signal, init_kernel=True, number_of_kernels=-1, kernel_frequencies=None,
                                          need_error_rate_fast=True, need_error_rate_accurate=False,
                                          need_reconstructed_signal=False, computation_mode=configuration.mode,
                                          ahp_period=configuration.ahp_period, ahp_high=configuration.ahp_high_value,
                                          selected_kernel_indexes=None,
                                          spiking_threshold=configuration.spiking_threshold,
                                          max_spike_count=configuration.max_spike_count,
                                          window_size=configuration.window_size,
                                          batch_size=configuration.window_spike_batch_size,
                                          snippet_len=configuration.snippet_length,
                                          overlap_len=configuration.signal_interleaving_length):
    """
    This method is invoked to reconstruct a signal which large enough
    so that it can't fit entirely in memory for processing
    :param batch_size:
    :param overlap_len:
    :param signal:
    :param init_kernel:
    :param number_of_kernels:
    :param kernel_frequencies:
    :param need_error_rate_fast:
    :param need_error_rate_accurate:
    :param need_reconstructed_signal:
    :param computation_mode:
    :param ahp_period:
    :param ahp_high:
    :param selected_kernel_indexes:
    :param spiking_threshold:
    :param max_spike_count:
    :param window_size:
    :param snippet_len:
    :return:
    """
    if init_kernel:
        kernel_manager.init(number_of_kernels, kernel_frequencies)
    total_len = len(signal)
    all_spikes = []
    all_spike_indexes = []
    all_thresholds = []
    total_signal_norm_square = 0
    each_kernel_spikes = [[] for i in range(number_of_kernels)]
    start_time = time.process_time()
    spike_generator.init()
    upsample_first = True
    if upsample_first:
        upsampled_full_signal = signal_utils.up_sample(signal)
        total_signal_norm_square = signal_utils.get_signal_norm_square(upsampled_full_signal)
    for i in range(math.ceil(total_len / snippet_len)):
        snippet_begin_time = max(i * snippet_len - overlap_len, 0)
        snippet_end_time = min(len(signal), (i + 1) * snippet_len)
        offset = snippet_begin_time * configuration.upsample_factor
        spike_start_time = min(len(signal), i * snippet_len) * configuration.upsample_factor
        if upsample_first:
            snippet = upsampled_full_signal[offset: snippet_end_time * configuration.upsample_factor]
        else:
            snippet = signal[snippet_begin_time:snippet_end_time]
            snippet = signal_utils.up_sample(snippet)
            total_signal_norm_square = total_signal_norm_square + \
                                       signal_utils.get_signal_norm_square(snippet[spike_start_time - offset:])

        signal_norm_sq, signal_kernel_convolutions = init_signal(snippet, computation_mode,
                                                                 False, selected_kernel_indexes)
        if configuration.variable_threshold and i == 0:
            max_vals = [np.max(a) if len(a) > 0 else 0 for a in signal_kernel_convolutions]
            max_conv = np.max(max_vals)

            while spiking_threshold > max_conv / 100:
                spiking_threshold = spiking_threshold / 10
            if configuration.debug:
                print(f'max convolution value: {max_conv} and spiking threshold: {spiking_threshold}')

        # total_signal_norm_square = total_signal_norm_square + signal_norm_sq
        spike_times, spike_indexes, threshold_values = spike_generator. \
            calculate_spike_times(signal_kernel_convolutions, ahp_period=ahp_period, ahp_high=ahp_high,
                                  selected_kernel_indexes=selected_kernel_indexes, threshold=spiking_threshold,
                                  offset=offset, spike_start_time=spike_start_time,
                                  end_time=-1 if i == (math.ceil(total_len / snippet_len) - 1)
                                  else snippet_end_time * configuration.upsample_factor,
                                  each_kernel_spikes=each_kernel_spikes)
        all_spikes = all_spikes + spike_times
        all_spike_indexes = all_spike_indexes + spike_indexes
        all_thresholds = all_thresholds + threshold_values
    threshold_error = -1
    if configuration.quantized_threshold_transmission:
        threshold_error = spike_generator.get_threshold_transmision_error_rate()
        if configuration.verbose:
            print(f'error in threshold transmission: {threshold_error}')
    if configuration.compute_time:
        print(f'time to compute all spikes: {time.process_time() - start_time}')
        start_time = time.process_time()
    recons_coeffs = None
    error_rate_fast = -1
    absolute_error_rate = -1
    recons = None

    if (0 < len(spike_times) < max_spike_count) \
            and (need_error_rate_fast or need_reconstructed_signal or need_error_rate_accurate):
        if configuration.windowing_batch_mode:
            recons_coeffs = calculate_reconstruction_in_batch_window_mode(all_spikes, all_spike_indexes,
                                                                          all_thresholds, window_size=window_size,
                                                                          batch_size=batch_size)
        else:
            recons_coeffs = calculate_reconstruction_in_window_mode(all_spikes, all_spike_indexes,
                                                                    all_thresholds, winddow_size=window_size)
        if configuration.compute_time:
            print(f'time to compute recons coeffs: {time.process_time() - start_time}')
        if need_error_rate_fast:
            error_rate_fast = calculate_reconstruction_error_rate_fast(recons_coeffs, all_thresholds,
                                                                       total_signal_norm_square)
        if (need_reconstructed_signal or need_error_rate_accurate) and recons_coeffs is not None:
            upsampled_full_signal = signal_utils.up_sample(signal)
            start_time = time.process_time()
            recons = get_reconstructed_signal(len(upsampled_full_signal),
                                              all_spikes, all_spike_indexes, recons_coeffs)
            if configuration.compute_time:
                print(f'time to compute reconstruction: {time.process_time() - start_time}')
            if configuration.debug:
                plot_utils.plot_function(recons, title='final full recons')
            absolute_error_rate = signal_utils.calculate_absolute_error_rate(upsampled_full_signal, recons)
            print(f'absolute error rate: {absolute_error_rate}')
    return all_spikes, all_spike_indexes, all_thresholds, recons_coeffs, \
           error_rate_fast, recons, absolute_error_rate, threshold_error, spiking_threshold


def drive_single_signal_reconstruction_iteratively(signal, init_kernel=True, number_of_kernels=-1,
                                                   kernel_frequencies=None,
                                                   need_error_rate_fast=True, need_error_rate_accurate=False,
                                                   need_reconstructed_signal=False, computation_mode=configuration.mode,
                                                   window_mode=False, window_size=-1, norm_threshold=0.01,
                                                   z_threshold=0.5, recompute_recons_coeff=True, show_z_vals=False,
                                                   signal_norm_sq=None, selected_kernel_indexes=None,
                                                   input_signal=None):
    """
    This method uses iterative technique to generate spikes and reconstruct a signal
    :param input_signal:
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
    sp_times, sp_indexes, thrs_values, recons_coeffs, z_scores, kernel_projections, \
    gamma_vals, recons_signal, gamma_vals_manual = iterative_spike_generator. \
        spike_and_reconstruct_iteratively(signal_kernel_convolutions, window_mode=window_mode,
                                          window_size=window_size,
                                          norm_threshold_for_new_spike=norm_threshold, z_thresholds=z_threshold,
                                          show_z_scores=show_z_vals, signal_norm_square=signal_norm_sq,
                                          selected_kernel_indexes=selected_kernel_indexes,
                                          input_signal=input_signal)
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
           signal_kernel_convolutions, z_scores, kernel_projections, gamma_vals, recons_signal, gamma_vals_manual

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
