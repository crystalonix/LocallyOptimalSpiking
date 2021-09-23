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
    audio_filepath = '../audio_text/'
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
    return reconstruction_manager.drive_single_signal_reconstruction(
        snippet, False, need_reconstructed_signal=need_reconstructed_signal)


sample_number = 5
sample_len = 1000
number_of_kernel = 100
norm_thrs = 0.01
print(f'all available frequencies are: '
      f'{gammatone_calculator.get_kernel_frequencies(gammatone_calculator.get_kernel_indexes(number_of_kernel))}')
kernel_bank1 = gammatone_calculator.get_kernel_indexes(number_of_kernel)

kernel_bank = kernel_bank1
kernel_manager.init(len(kernel_bank), kernel_bank)
# this_snippet, this_norm, fl_len, fl_signal, fl_norm = get_first_snippet_above_threshold_norm(
#     sample_number, sample_len, norm_threshold=0.01)
ind1 = 5
ind2 = 70
delta = 200
vector_ip_val = kernel_manager.calculate_kernel_ip_from_bspline_efficient(index1=ind1, index2=ind2, time_delta=delta)
manual_ip_val = kernel_manager.calculate_kernel_ip_from_bspline(index1=ind1, index2=ind2, time_delta=delta)
print(f'{vector_ip_val}:vector ip, {manual_ip_val}: manual ip value')
# print(f'manual ip:{ip}')
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
