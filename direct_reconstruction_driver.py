import configuration
import kernel_manager
import reconstruction_driver
import signal_utils
import reconstruction_manager
import plot_utils
import numpy as np
import wav_file_handler

sample_numbers = [i for i in range(7, 20)]
# [i for i in range(9, 30)]
sample_len = 15000
snip_len = 10000
overlap = 5000
# choosing approx 5s snippets
full_signal_len = 250000
number_of_kernel = 20
select_kernel_indexes = [i for i in range(2, number_of_kernel)]
signal_norm_thrs = -1.0
# 1e-4
spiking_thresholds = np.array([5e-6])
# [5e-5, 5e-6, 5e-7]
upsample_factor = configuration.upsample_factor
ahp_periods = np.array([500]) * upsample_factor
# np.array([50, 100, 200, 500, 1000.0, 2000.0]) * upsample_factor
ahp_highs = np.array([10]) * upsample_factor
# np.array([1e-1, 1, 10, 100]) * upsample_factor
max_spike = 2500000
#           1000000
# [5e-3, 2e-3, 5e-4, 2e-4, 5e-5, 2e-5, 5e-6, 2e-6, 5e-7, 2e-7, 5e-8, 2e-8, 5e-9, 5e-10]
win_mode = True
win_size = 1000
reconstruct_full_signal = True
show_plots = True
need_recons = True
include_stats = True
save_recons_to_wav = False
reconstruction_stats = []
reports_file = 'all_reconstruction_log1.txt'
signal_from_wav_file = False
kernel_manager.init(number_of_kernel)
# for i in range(number_of_kernel):
#     print(f'len of kernel: {len(kernel_manager.all_kernels[i])}')
i = 0
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
        for ahp_period in ahp_periods:
            for ahp_high in ahp_highs:
                ahp_high = ahp_high * spiking_threshold
                i = i + 1
                if reconstruct_full_signal:
                    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                    reconstruction, abs_error = reconstruction_manager.drive_piecewise_signal_reconstruction(
                        full_signal, False, number_of_kernels=number_of_kernel,
                        need_reconstructed_signal=need_recons, ahp_period=ahp_period,
                        selected_kernel_indexes=select_kernel_indexes, spiking_threshold=spiking_threshold,
                        ahp_high=ahp_high, max_spike_count=max_spike,
                        window_size=win_size, snippet_len=snip_len, overlap_len=overlap)
                else:
                    spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, reconstruction = \
                        reconstruction_manager.drive_single_signal_reconstruction(
                            snippet, False, number_of_kernels=number_of_kernel, need_reconstructed_signal=need_recons,
                            ahp_period=ahp_period, selected_kernel_indexes=select_kernel_indexes,
                            spiking_threshold=spiking_threshold,
                            signal_norm_square=signal_norm_square,
                            signal_kernel_convolutions=signal_kernel_convolutions,
                            ahp_high=ahp_high, max_spike_count=max_spike, window_mode=win_mode, window_size=win_size)
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
                print(f'snippet#: {sample_number}, spike rate: '
                      f'{configuration.actual_sampling_rate * len(spike_times) / sample_len}, '
                      f'reconstruction error: {error_rate}, '
                      f'number of spikes: {len(spike_times)} threshold: {spiking_threshold}, ahp high:{ahp_high}',
                      f'ahp period:{ahp_period}')
                if include_stats:
                    reconstruction_stats.append([sample_number, error_rate,
                                                 configuration.actual_sampling_rate * len(spike_times) / sample_len,
                                                 spiking_threshold, ahp_high, ahp_period])
                # TODO: use it for final run
                # if len(spike_times) > max_spike:
                #     break
np.savetxt(reports_file, np.array(reconstruction_stats))
