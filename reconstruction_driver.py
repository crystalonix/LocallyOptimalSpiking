import math

import numpy as np

import configuration
import file_utils
import plot_utils
import reconstruction_manager
import gammatone_calculator
import signal_utils
import kernel_manager


def split_signal_into_snippets(signal_index, sample_length, norm_threshold=0):
    full_signal, full_norm = get_signal(signal_index)
    full_len = len(full_signal)
    number_of_snippets = math.ceil(full_len / sample_length)
    all_snippets = []
    all_norms = []
    for i in range(number_of_snippets):
        this_snippet = full_signal[i * sample_length: min((i + 1) * sample_length, full_len)]
        this_norm = np.sqrt(signal_utils.get_signal_norm_square(this_snippet, configuration.actual_sampling_rate))
        if this_norm >= norm_threshold:
            all_snippets.append(this_snippet)
            all_norms.append(this_norm)
    return all_norms, all_snippets, full_len, full_signal, full_norm


def get_signal(signal_index):
    audio_filepath = '../../audio_text/'
    file_name = audio_filepath + str(signal_index) + '.txt'
    print(file_name)
    full_signal = file_utils.read_1D_np_array(file_name)
    return full_signal, np.sqrt(signal_utils.get_signal_norm_square(full_signal, configuration.actual_sampling_rate))


def get_first_snippet_above_threshold_norm(signal_index, sample_length, norm_threshold):
    full_signal, full_norm = get_signal(signal_index)
    full_len = len(full_signal)
    number_of_snippets = math.ceil(full_len / sample_length)
    for i in range(number_of_snippets):
        this_snippet = full_signal[i * sample_length: min((i + 1) * sample_length, full_len)]
        this_norm = np.sqrt(signal_utils.get_signal_norm_square(this_snippet, configuration.actual_sampling_rate))
        if this_norm > norm_threshold:
            return this_snippet, this_norm, full_len, full_signal, full_norm
    return None, -1, full_len, full_signal, full_norm


def drive_full_signal_reconstruction(signal_index, sample_length):
    pass


def drive_select_snippet_reconstruction(signal_index, sample_length, norm_threshold=0.0,
                                        need_reconstructed_signal=False):
    snippet, norm, _, _, _ = get_first_snippet_above_threshold_norm(signal_index, sample_length, norm_threshold)
    snippet = signal_utils.upsample(snippet)
    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction = \
        reconstruction_manager.drive_single_signal_reconstruction(
            snippet, False, need_reconstructed_signal=need_reconstructed_signal)
    return snippet, spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction


def drive_select_snippet_reconstruction_iteratively(signal_index, sample_length, norm_threshold=0.0,
                                                    need_reconstructed_signal=False, window_mode=False, window_size=-1,
                                                    ip_threshold=0.001, rectify_coefficients=False):
    snippet, norm, _, _, _ = get_first_snippet_above_threshold_norm(signal_index, sample_length, norm_threshold)
    snippet = signal_utils.upsample(snippet)
    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction = \
        reconstruction_manager.drive_single_signal_reconstruction_iteratively(
            snippet, False, need_reconstructed_signal=need_reconstructed_signal, window_mode=window_mode,
            window_size=window_size, ip_threshold=ip_threshold, recompute_recons_coeff=rectify_coefficients)
    return snippet, spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction


sample_number = 5
sample_len = 10000
number_of_kernel = 10
norm_thrs = 0.00001
win_mode = True
win_size = 30
ip_thrs = 0.9
win_sizes = [30
             # ,20, 30
             # ,40
             ]
ip_thresholds = [
    # [0.9, 0.95, 0.995, 0.9995],
    # [0.8, 0.9, 0.95, 0.99, 0.999],
    list(np.arange(0.5, 0.9, 0.05))
    # [0.7, 0.8, 0.9, 0.95, 0.99]
    # ,[0.5, 0.6, 0.7, 0.8, 0.9]
]
rectify_coeffs = True
single_snippet = False
# configuration.ahp_period = configuration.ahp_period/2.0
# print(f'all available frequencies are: '
#       f'{gammatone_calculator.get_kernel_frequencies(gammatone_calculator.get_kernel_indexes(number_of_kernel))}')
# kernel_bank1 = gammatone_calculator.get_kernel_indexes(number_of_kernel)
#
# kernel_bank = kernel_bank1
# kernel_manager.init(len(kernel_bank), kernel_bank)
# # this_snippet, this_norm, fl_len, fl_signal, fl_norm = get_first_snippet_above_threshold_norm(
# # sample_number, sample_len, norm_threshold=0.01)
# ind1 = 5
# ind2 = 70
# delta = 200
# vector_ip_val = kernel_manager.calculate_kernel_ip_from_bspline_efficient(index1=ind1, index2=ind2, time_delta=delta)
# manual_ip_val = kernel_manager.calculate_kernel_ip_from_bspline(index1=ind1, index2=ind2, time_delta=delta)
# print(f'{vector_ip_val}:vector ip, {manual_ip_val}: manual ip value')
kernel_manager.init(number_of_kernel)
########################################################################################
####################### iterative signal reconstruction ################################
########################################################################################
if not single_snippet:
    print(f'to start iterative reconstruction')
    # TODO: uncomment this for a grid based run
    all_errors = []
    all_recons = []
    sp_times_len = []
    legends = []
    i = -1
    for w in win_sizes:
        tmp = []
        sp_tmp = []
        i = i + 1
        for i_th in ip_thresholds[i]:
            print(f'running for window size:{w} and threshold:{i_th}')
            _, sp_times, _, _, _, error_rate_fast_itr, recons_itr = \
                drive_select_snippet_reconstruction_iteratively(sample_number, sample_len, norm_threshold=norm_thrs,
                                                                need_reconstructed_signal=True, window_mode=win_mode,
                                                                window_size=w, ip_threshold=i_th,
                                                                rectify_coefficients=rectify_coeffs)
            tmp.append(error_rate_fast_itr)
            sp_tmp.append(len(sp_times))

        all_errors.append(tmp)
        sp_times_len.append(sp_tmp)
        legends.append(f'error for window size: {w}')
    plot_utils.plot_functions_in_one_plot(all_errors, sp_times_len, legends=legends)

# TODO: uncomment for single snippet recons
if (single_snippet):
    this_signal, sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons = \
        drive_select_snippet_reconstruction_iteratively(sample_number, sample_len, norm_threshold=norm_thrs,
                                                        need_reconstructed_signal=True, window_mode=win_mode,
                                                        window_size=win_size, ip_threshold=ip_thrs,
                                                        rectify_coefficients=rectify_coeffs)
    print(f'iterative recons error rate is: {error_rate_fast} and number of spike: {len(sp_indexes)}')
    # plot_utils.plot_function(recons, title='iterative reconstruction')
    _, sp_times_1, sp_indexes_1, _, recons_coeffs_1, error_rate_fast_1, recons_1 = \
        drive_select_snippet_reconstruction(sample_number, sample_len, norm_threshold=norm_thrs,
                                            need_reconstructed_signal=True)
    # plot_utils.plot_function(recons_1, title='Normal reconstruction')
    print(f'batch recons error rate is: {error_rate_fast_1} and number of spike: {len(sp_indexes_1)}')
    # plot_utils.plot_functions_in_one_plot([this_signal, recons, recons_1])

    # plot_utils.plot_function(this_signal, title='original signal')
    plot_utils.plot_functions([this_signal, recons, recons_1],
                              plot_titles=['original signal',
                                           f'iterative reconstruction with {len(sp_times)} spikes and error rate: {error_rate_fast}',
                                           f'normal reconstruction with{len(sp_indexes_1)} spikes and error rate:{error_rate_fast_1}'],
                              x_ticks_on=False)
    # plot_utils.plot_function(recons)
    print('we are done here')
########################################################################################
####################### bring this back after checking ################################
########################################################################################

# this_signal, sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons = \
#     drive_select_snippet_reconstruction(sample_number, sample_len, norm_threshold=norm_thrs)
# print(f'error rate is: {error_rate_fast}')
# plot_utils.plot_function(this_signal)


# drive_full_signal_reconstruction(sample_number, sample_len)
# print(f'The norms of are: {norms}, number of snippets: {len(norms)}, '
#       f'total signal norm: {fl_norm} and the full length of signal: {fl_len}')
# plot_utils.plot_function(norms)

# kernel_manager.init(number_of_kernel, mode='compressed', normalize=True)
# kernel_manager.init(number_of_kernel, mode='expanded', normalize=True)
# x, kernel_convs = reconstruction_manager.init_signal(test_signal, mode='compressed')
#
# kernel_manager.init(number_of_kernel, mode='compressed', normalize=True, load_from_cache=True)
# x1, kernel_convs1 = init_signal(test_signal, mode='compressed')
#
# id = 2
# f1 = kernel_convs[id]
# f2 = kernel_convs1[id]
# plot_utils.plot_function(f1, title='compressed signal kernel conv')
# plot_utils.plot_function(f2, title='compressed signal kernel conv from cache')
# ######################################## Uncomment me ###########################################
# sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons = drive_single_signal_reconstruction(
#     test_signal, init_kernel=False, need_error_rate_fast=False, need_reconstructed_signal=False)
# print(f'number of generated spikes: {len(sp_times)}')
# plot_utils.spike_train_plot(sp_times, sp_indexes)
