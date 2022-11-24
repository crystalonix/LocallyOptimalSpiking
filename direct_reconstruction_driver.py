import math
import time

import numpy as np

import configuration
import file_utils
import kernel_manager
import plot_utils
import reconstruction_driver
import reconstruction_manager
import signal_utils
import wav_file_handler

sample_numbers = [i for i in range(400, 600)]
# [i for i in range(9, 30)]
sample_len = 100000
snip_len = 10000
overlap = 7000
# choosing approx 5s snippets
full_signal_len = 100000
number_of_kernel = 50
# exclude some of the very low frequency kernel to make it computationally efficient
select_kernel_indexes = [i for i in range(math.ceil(number_of_kernel / 10), number_of_kernel)]
signal_norm_thrs = -1.0
# 1e-4
spiking_thresholds = np.array([5e-5])
# [5e-5, 5e-6, 5e-7]
upsample_factor = configuration.upsample_factor
# arrange the ahp periods in a systematic way so that in tunes the firing rate appropriately
ahp_periods = np.array(range(1000, 100, -100)) * configuration.upsample_factor
ahp_periods = np.concatenate((ahp_periods, np.array(range(100, 20, -20)) * configuration.upsample_factor))
# np.array([1000.0, 500, 200, 100]) * upsample_factor
# np.array([50, 100, 200, 500, 1000.0, 2000.0]) * upsample_factor
ahp_highs = np.array([1]) * upsample_factor
# np.array([1e-1, 1, 10, 100]) * upsample_factor
max_spike = full_signal_len / 1.5
#           1000000
# [5e-3, 2e-3, 5e-4, 2e-4, 5e-5, 2e-5, 5e-6, 2e-6, 5e-7, 2e-7, 5e-8, 2e-8, 5e-9, 5e-10]
win_mode = True
win_factor = 50e6
max_win_size = 15000
spike_batch_size = 500
reconstruct_full_signal = True
show_plots = False
need_recons = False
accurate_err = True
save_recons_to_wav = False
reconstruction_stats = []
stats_csv_file = 'recons_reports_on_40_kernel.csv'
signal_from_wav_file = False
kernel_manager.init(number_of_kernel)
for i in range(number_of_kernel):
    print(f'len of kernel: {len(kernel_manager.all_kernels[i])}')
i = 0
snapshot_interval = 1
reconstruction_stats = []
for sample_number in sample_numbers:
    full_signal = reconstruction_driver.get_signal(sample_number, read_from_wav=signal_from_wav_file)
    if full_signal_len > - 1:
        full_signal = full_signal[:full_signal_len]
        actual_signal = full_signal
    if not reconstruct_full_signal:
        snippet, norm, _, _, _ = reconstruction_driver. \
            get_first_snippet_above_threshold_norm(full_signal, sample_len, signal_norm_thrs)
        actual_signal = snippet
        snippet = signal_utils.up_sample(snippet)
        signal_norm_square, signal_kernel_convolutions = reconstruction_manager.init_signal(snippet, configuration.mode)
    for spiking_threshold in spiking_thresholds:
        for ahp_high in ahp_highs:
            ahp_high = ahp_high * spiking_threshold
            for ahp_period in ahp_periods:
                i = i + 1
                threshold_error = -1
                win_size = min(max_win_size, int(win_factor / ahp_period))
                print(f'win size:{win_size}')
                if configuration.compute_time:
                    initial_time = time.time()
                if reconstruct_full_signal:
                    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                    reconstruction, abs_error, threshold_error, spiking_threshold = \
                        reconstruction_manager.drive_piecewise_signal_reconstruction(
                            actual_signal, False, number_of_kernels=number_of_kernel,
                            need_reconstructed_signal=need_recons, ahp_period=ahp_period,
                            selected_kernel_indexes=select_kernel_indexes, spiking_threshold=spiking_threshold,
                            ahp_high=ahp_high, max_spike_count=max_spike, need_error_rate_accurate=accurate_err,
                            window_size=win_size, snippet_len=snip_len,
                            overlap_len=overlap, batch_size=spike_batch_size)
                else:
                    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                    reconstruction, abs_error = reconstruction_manager.drive_single_signal_reconstruction(
                        snippet, False, number_of_kernels=number_of_kernel, need_reconstructed_signal=need_recons,
                        ahp_period=ahp_period, selected_kernel_indexes=select_kernel_indexes,
                        spiking_threshold=spiking_threshold, need_error_rate_accurate=accurate_err,
                        signal_norm_square=signal_norm_square,
                        signal_kernel_convolutions=signal_kernel_convolutions,
                        ahp_high=ahp_high, max_spike_count=max_spike, window_mode=win_mode, window_size=win_size)
                time_diff = -1
                if configuration.compute_time:
                    time_diff = time.time() - initial_time
                    print(f'time for this iteration: {time_diff}')
                reconstruction_stats.append([sample_number, abs_error, error_rate, threshold_error,
                                             len(spike_times) / len(actual_signal), ahp_period, ahp_high,
                                             spiking_threshold, time_diff, win_size])
                if configuration.debug:
                    print(f'all spikes occurring at: {spike_times}')
                if save_recons_to_wav and reconstruction is not None:
                    signal_to_save = signal_utils.down_sample(reconstruction)
                    file = configuration.training_sub_sample_folder_path + 'reconstruction-' \
                           + str(sample_number) + '.wav'
                    wav_file_handler.store_float_date_to_wav(file, signal_to_save)
                if show_plots and reconstruction is not None:
                    plot_utils.plot_functions([signal_utils.down_sample(reconstruction, up_factor=10), actual_signal],
                                              plot_titles=[f'normal reconstruction with{len(spike_times)}'
                                                           f' spikes and error rate:{error_rate}', f'original signal'])
                    plot_utils.spike_train_plot(spike_times, spike_indexes,
                                                title=f'all spikes for signal {sample_number}')
                len_of_signal = full_signal_len if reconstruct_full_signal else sample_len
                print(f'snippet#: {sample_number}, spike rate: '
                      f'{configuration.actual_sampling_rate * len(spike_times) / len_of_signal}, '
                      f'reconstruction error: {error_rate}, '
                      f'number of spikes: {len(spike_times)} threshold: {spiking_threshold}, ahp high:{ahp_high}',
                      f'ahp period:{ahp_period}')
                if i % snapshot_interval == 0:
                    file_utils.write_array_to_csv(filename=stats_csv_file, data=reconstruction_stats)
                if len(spike_times) > max_spike:
                    break
file_utils.write_array_to_csv(filename=stats_csv_file, data=reconstruction_stats)

###################################################################################
############################ used for a reconstruction demo #######################
###################################################################################
# sample_number = 5
# full_signal = reconstruction_driver.get_signal(sample_number, read_from_wav=signal_from_wav_file)
# snippet, norm, _, _, _ = reconstruction_driver. \
#     get_first_snippet_above_threshold_norm(full_signal, sample_len, signal_norm_thrs)
# actual_signal = snippet
# snippet = signal_utils.up_sample(snippet)
# snippet = snippet[: 40000]
# signal_norm_square, signal_kernel_convolutions = reconstruction_manager.init_signal(snippet, configuration.mode)
# plot_utils.plot_function(snippet)
# selected_index=5
# select_kernel_indexes = [selected_index]
# all_thresholds = [[] for i in range(number_of_kernel)]
# spike_times, spike_indexes, threshold_values = \
#     spike_generator.calculate_spikes_for_all_kernels(signal_kernel_convolutions, select_kernel_indexes, ahp_period=4000,
#                                                      ahp_high=5e-6, threshold=5e-7, all_threshold=all_thresholds)
# plot_utils.plot_functions_in_one_plot([signal_kernel_convolutions[selected_index],
#                                        all_thresholds[selected_index]], colors=['blue', 'green'])
# print("done")
