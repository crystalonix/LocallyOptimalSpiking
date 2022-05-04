import math

import numpy as np

import configuration
import file_utils
import plot_utils
import reconstruction_manager
import gammatone_calculator
import signal_utils
import kernel_manager
import matplotlib.pyplot as plt
import common_utils


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
                                        need_reconstructed_signal=False, ahp_period=configuration.ahp_period,
                                        selected_kernel_indexes=None):
    snippet, norm, _, _, _ = get_first_snippet_above_threshold_norm(signal_index, sample_length, norm_threshold)
    snippet = signal_utils.upsample(snippet)
    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction = \
        reconstruction_manager.drive_single_signal_reconstruction(
            snippet, False, need_reconstructed_signal=need_reconstructed_signal, ahp_period=ahp_period,
            selected_kernel_indexes=selected_kernel_indexes)
    return snippet, spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction


def drive_select_snippet_reconstruction_iteratively(signal_index, sample_length, norm_threshold=0.0,
                                                    need_reconstructed_signal=False, window_mode=False, window_size=-1,
                                                    new_kernel_norm_threshold=0.001, z_threshold=0.5,
                                                    rectify_coefficients=False,
                                                    show_z_vals=False, test_signal=None, selected_kernel_indexes=None):
    if test_signal is not None:
        snippet = test_signal
        norm = signal_utils.get_signal_norm_square(snippet, samp_rate=configuration.actual_sampling_rate)
        norm = np.sqrt(norm)
    else:
        snippet, norm, _, _, _ = get_first_snippet_above_threshold_norm(signal_index, sample_length, norm_threshold)
    snippet = signal_utils.upsample(snippet)
    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction, \
    all_convs, z_scores, kernel_projections, gamma_vals = reconstruction_manager.drive_single_signal_reconstruction_iteratively(
        snippet, False, need_reconstructed_signal=need_reconstructed_signal, window_mode=window_mode,
        window_size=window_size, norm_threshold=new_kernel_norm_threshold, recompute_recons_coeff=rectify_coefficients,
        show_z_vals=show_z_vals, z_threshold=z_threshold, signal_norm_sq=norm * norm,
        selected_kernel_indexes=selected_kernel_indexes)
    return snippet, spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, \
           error_rate, reconstruction, all_convs, z_scores, kernel_projections, gamma_vals


ahp = configuration.ahp_period / 1.0
sample_numbers = [5]
# [i for i in range(5, 10)]
sample_number = 6
sample_len = 10000
number_of_kernel = 2
select_kernel_indexes = [0]
signal_norm_thrs = 1e-4
win_mode = False
win_size = 30
norm_thres = 1e-3
z_thrs = np.array([2e3, 0.001]) * 1e-7  # 1e-8
# z_thrs = np.array([0.0001, .001, .01, .1, 1.0]) * 1e-6
win_sizes = [50
             # ,20, 30
             # ,40
             ]
ip_thresholds = [
    # [0.9, 0.95, 0.995, 0.9995],
    # [0.8, 0.9, 0.95, 0.99, 0.999],
    list(np.arange(0.5, 0.7, 0.02))
    # [0.7, 0.8, 0.9, 0.95, 0.99]
    # ,[0.5, 0.6, 0.7, 0.8, 0.9]
]
iterative_recons = True
direct_recons = True
rectify_coeffs = True
single_snippet = True
show_z = False
turn_on_recons_plot = True
show_z_scores_plot = True
club_z_score_with_proj_thrs = False

params_grid = {
    'sample_number': [i for i in range(5, 10)],
}
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
            _, sp_times, _, _, _, error_rate_fast_itr, recons_itr, all_convs, z_scores, kernel_projections, gamma_vals \
                = drive_select_snippet_reconstruction_iteratively(sample_number, sample_len,
                                                                  norm_threshold=signal_norm_thrs,
                                                                  need_reconstructed_signal=True, window_mode=win_mode,
                                                                  window_size=w, new_kernel_norm_threshold=i_th,
                                                                  rectify_coefficients=rectify_coeffs,
                                                                  show_z_vals=show_z_scores_plot,
                                                                  z_threshold=z_thrs)
            tmp.append(error_rate_fast_itr)
            sp_tmp.append(len(sp_times))

        all_errors.append(tmp)
        sp_times_len.append(sp_tmp)
        legends.append(f'error for window size: {w}')
    plot_utils.plot_functions_in_one_plot(all_errors, sp_times_len, legends=legends)

# TODO: uncomment for single snippet recons
if single_snippet:
    if iterative_recons:
        params_dict = {}
        for i in range(len(sample_numbers)):
            sample_number = sample_numbers[i]
            this_signal, sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, \
            recons, all_convs, z_vals, kernel_projections, gamma_vals = \
                drive_select_snippet_reconstruction_iteratively(sample_number, sample_len,
                                                                norm_threshold=signal_norm_thrs,
                                                                need_reconstructed_signal=True, window_mode=win_mode,
                                                                window_size=win_size,
                                                                new_kernel_norm_threshold=norm_thres,
                                                                rectify_coefficients=rectify_coeffs,
                                                                show_z_vals=show_z_scores_plot,
                                                                z_threshold=z_thrs,
                                                                selected_kernel_indexes=select_kernel_indexes)
            print(f'iterative recons error rate is: {error_rate_fast} and number of spike: {len(sp_indexes)}')
            print(f'model description:- z_threshold {z_thrs}, number of kernels: {number_of_kernel}, '
                  f'schur power: {configuration.SCHUR_POWER}')
            common_utils.sort_spikes_on_kernel_indexes(spike_times=sp_times, spike_indexes=sp_indexes,
                                                       num_kernels=number_of_kernel)
            # plot_utils.plot_function(recons, title='iterative reconstruction')

    if direct_recons:
        _, sp_times_1, sp_indexes_1, _, recons_coeffs_1, error_rate_fast_1, recons_1 = \
            drive_select_snippet_reconstruction(sample_number, sample_len, norm_threshold=signal_norm_thrs,
                                                need_reconstructed_signal=True, ahp_period=ahp,
                                                selected_kernel_indexes=select_kernel_indexes)
    if not iterative_recons:
        plot_utils.plot_function(recons_1,
                                 title=f'normal reconstruction with{len(sp_indexes_1)}'
                                       f' spikes and error rate:{error_rate_fast_1}')
        print(f'batch recons error rate is: {error_rate_fast_1} and number of spike: {len(sp_indexes_1)}')
    # plot_utils.plot_functions_in_one_plot([this_signal, recons, recons_1])

    # plot_utils.plot_function(this_signal, title='original signal')
    if iterative_recons and turn_on_recons_plot:
        plot_utils.plot_functions([this_signal, recons, recons_1],
                                  plot_titles=['original signal',
                                               f'iterative reconstruction with {len(sp_times)} spikes and error rate: {error_rate_fast}',
                                               f'normal reconstruction with{len(sp_indexes_1)} spikes and error rate:{error_rate_fast_1}'],
                                  x_ticks_on=False)
        plot_utils.plot_multiple_spike_trains([sp_times, sp_times_1], [sp_indexes, sp_indexes_1], plot_titles=[
            f'spikes plot for iterative reconstruction with total {len(sp_times)}spikes',
            f'spikes plot for normal reconstruction with total {len(sp_times_1)}spikes'])
        if show_z:
            plot_utils.plot_functions(z_vals + all_convs,
                                      plot_titles=['convs n z_scores' for i in range(len(z_vals) + len(all_convs))])
    if show_z_scores_plot:
        for this_index in select_kernel_indexes:
            plot_utils.plot_kernel_spike_profile(sp_times[sp_indexes == this_index], all_convs[this_index],
                                                 z_vals[this_index], kernel_projections[this_index], gamma_vals[this_index],
                                                 club_z_score_threshold=club_z_score_with_proj_thrs,
                                                 kernel_index=this_index)

        plt.show()
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
