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
import logging
import spike_generator
from scipy.fft import fft, fftfreq

# initialize the logger
logging.basicConfig(filename=configuration.log_file, level=configuration.logging_level)
single_snippet_mode = 1
batch_mode = 2
demo_mode = 3
large_experiment_mode = 4
csc_experiment_mode = 5
mode_for_running_this_driver = large_experiment_mode
# large_experiment_mode, csc_experiment_mode, demo_mode
snip_len = 10000
overlap = 7000
if mode_for_running_this_driver == large_experiment_mode or mode_for_running_this_driver == csc_experiment_mode:
    ##################### uncomment for large set of experiments #########################
    sample_numbers = [i for i in range(1, 200)]
    # [i for i in range(9, 30)]
    sample_lens = [100000]
    # choosing approx 5s snippets
    full_signal_len = 100000
    number_of_kernel = 20
    # exclude some of the very low frequency kernels to make it computationally efficient
    select_kernel_indexes = [i for i in range(math.ceil(number_of_kernel / 10), number_of_kernel)]
    signal_norm_thrs = -1.0
    # 1e-4
    spiking_thresholds = np.array([5e-6])
    # [5e-5, 5e-6, 5e-7]
    upsample_factor = configuration.upsample_factor
    # arrange the ahp periods in a systematic way so that it tunes the firing rate appropriately
    ahp_periods = np.array(range(1000, 100, -100)) * configuration.upsample_factor
    ahp_periods = np.concatenate((ahp_periods, np.array(range(100, 20, -20)) * configuration.upsample_factor))
    # np.array([1000.0, 500, 200, 100]) * upsample_factor
    # np.array([50, 100, 200, 500, 1000.0, 2000.0]) * upsample_factor
    ahp_highs = np.array([10]) * upsample_factor
    # np.array([1e-1, 1, 10, 100]) * upsample_factor
    max_spike = full_signal_len / 1.5
    #           1000000
    # [5e-3, 2e-3, 5e-4, 2e-4, 5e-5, 2e-5, 5e-6, 2e-6, 5e-7, 2e-7, 5e-8, 2e-8, 5e-9, 5e-10]
    win_mode = True
    win_factor = 1e8
    max_win_size = 25000
    spike_batch_size = 500
    reconstruct_full_signal = True
    reconstruct_with_lateral_inhibition = True
    show_plots = False
    need_recons = False
    accurate_err = True
    save_recons_to_wav = False
    reconstruction_stats = []
    stats_csv_file = 'recons_reports_on_80_kernel.csv'
    signal_from_wav_file = False

    ############## config for small set of experiments for CSC comparison ###################
    if mode_for_running_this_driver == csc_experiment_mode:
        sample_numbers = [i for i in range(1, 20)]
        # [i for i in range(9, 30)]
        sample_lens = [20000, 30000, 40000, 50000, 60000]
        overlap = 7000
        # choosing approx 5s snippets
        number_of_kernel = 10
        # exclude some of the very low frequency kernel to make it computationally efficient
        select_kernel_indexes = [i for i in range(math.ceil(number_of_kernel / 10), number_of_kernel)]
        signal_norm_thrs = -1.0
        # 1e-4
        spiking_thresholds = np.array([5e-6])
        # [5e-5, 5e-6, 5e-7]
        upsample_factor = configuration.upsample_factor
        # arrange the ahp periods in a systematic way so that in tunes the firing rate appropriately
        ahp_periods = np.array(range(1000, 100, -300)) * configuration.upsample_factor
        ahp_periods = np.concatenate((ahp_periods, np.array(range(100, 0, -10)) * configuration.upsample_factor))
        # ahp_periods = np.concatenate((ahp_periods, np.array(range(20, 0, -4)) * configuration.upsample_factor))
        # np.array([1000.0, 500, 200, 100]) * upsample_factor
        # np.array([50, 100, 200, 500, 1000.0, 2000.0]) * upsample_factor
        ahp_highs = np.array([10]) * upsample_factor
        # np.array([1e-1, 1, 10, 100]) * upsample_factor

        #           1000000
        # [5e-3, 2e-3, 5e-4, 2e-4, 5e-5, 2e-5, 5e-6, 2e-6, 5e-7, 2e-7, 5e-8, 2e-8, 5e-9, 5e-10]
        win_mode = True
        win_factor = 1e6 * number_of_kernel
        max_win_size = 10000
        spike_batch_size = 500
        reconstruct_full_signal = True
        reconstruct_with_lateral_inhibition = True
        show_plots = False
        need_recons = False
        accurate_err = True
        save_recons_to_wav = False
        reconstruction_stats = []
        stats_csv_file = 'recons_reports_csc_comparison_10_kernel.csv'
        signal_from_wav_file = False

    kernel_manager.init(number_of_kernel)
    # for i in range(number_of_kernel):
    #     print(f'len of kernel: {len(kernel_manager.all_kernels[i])}')
    i = 0
    snapshot_interval = 1
    reconstruction_stats = []
    for sample_len in sample_lens:
        full_signal_len = sample_len
        # TODO: for small experiments uncomemment the following line
        # snip_len = sample_len
        max_spike = int(full_signal_len * 0.8)
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
                signal_norm_square, signal_kernel_convolutions = reconstruction_manager.init_signal(snippet,
                                                                                                    configuration.mode)
            for spiking_threshold in spiking_thresholds:
                for ahp_high in ahp_highs:
                    ahp_high = ahp_high * spiking_threshold
                    for ahp_period in ahp_periods:
                        i = i + 1
                        threshold_error = -1
                        win_size = min(max_win_size, int(win_factor / ahp_period))
                        logging.debug(f'win size:{win_size}')
                        if configuration.compute_time:
                            initial_time = time.time()
                        if reconstruct_full_signal and not reconstruct_with_lateral_inhibition:
                            spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                            reconstruction, abs_error, threshold_error, spiking_threshold = \
                                reconstruction_manager.drive_piecewise_signal_reconstruction(
                                    actual_signal, False, number_of_kernels=number_of_kernel,
                                    need_reconstructed_signal=need_recons, ahp_period=ahp_period,
                                    selected_kernel_indexes=select_kernel_indexes, spiking_threshold=spiking_threshold,
                                    ahp_high=ahp_high, max_spike_count=max_spike, need_error_rate_accurate=accurate_err,
                                    window_size=win_size, snippet_len=snip_len,
                                    overlap_len=overlap, batch_size=spike_batch_size)
                        elif reconstruct_with_lateral_inhibition:
                            spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                            reconstruction, abs_error, threshold_error, spiking_threshold = \
                                reconstruction_manager.drive_signal_reconstruction_with_lateral_inhibition_parallel(
                                    actual_signal, False, number_of_kernels=number_of_kernel,
                                    need_reconstructed_signal=need_recons, ahp_period=ahp_period,
                                    selected_kernel_indexes=select_kernel_indexes, spiking_threshold=spiking_threshold,
                                    ahp_high=ahp_high, max_spike_count=max_spike, need_error_rate_accurate=accurate_err,
                                    window_size=win_size, snippet_len=snip_len,
                                    overlap_len=overlap, batch_size=spike_batch_size)
                        else:
                            spike_times, spike_indexes, thrshold_values, reconstruction_coefficients, error_rate, \
                            reconstruction, abs_error = reconstruction_manager.drive_single_signal_reconstruction(
                                snippet, False, number_of_kernels=number_of_kernel,
                                need_reconstructed_signal=need_recons,
                                ahp_period=ahp_period, selected_kernel_indexes=select_kernel_indexes,
                                spiking_threshold=spiking_threshold, need_error_rate_accurate=accurate_err,
                                signal_norm_square=signal_norm_square,
                                signal_kernel_convolutions=signal_kernel_convolutions,
                                ahp_high=ahp_high, max_spike_count=max_spike, window_mode=win_mode,
                                window_size=win_size)
                        time_diff = -1
                        if configuration.compute_time:
                            time_diff = time.time() - initial_time
                            logging.debug(f'time for this iteration: {time_diff}')
                        reconstruction_stats.append([sample_number, abs_error, error_rate, threshold_error,
                                                     len(spike_times) / len(actual_signal), ahp_period, ahp_high,
                                                     spiking_threshold, time_diff, win_size, full_signal_len])
                        if configuration.debug:
                            print(f'all spikes occurring at: {spike_times}')
                        if save_recons_to_wav and reconstruction is not None:
                            signal_to_save = signal_utils.down_sample(reconstruction)
                            file = configuration.training_sub_sample_folder_path + 'reconstruction-' \
                                   + str(sample_number) + '.wav'
                            wav_file_handler.store_float_date_to_wav(file, signal_to_save)
                        if show_plots and reconstruction is not None:
                            plot_utils.plot_functions(
                                [signal_utils.down_sample(reconstruction, up_factor=10), actual_signal],
                                plot_titles=[f'normal reconstruction with{len(spike_times)}'
                                             f' spikes and error rate:{error_rate}', f'original signal'])
                            plot_utils.spike_train_plot(spike_times, spike_indexes,
                                                        title=f'all spikes for signal {sample_number}')
                        len_of_signal = full_signal_len if reconstruct_full_signal else sample_len
                        logging.debug(f'snippet#: {sample_number}, spike rate: '
                                      f'{configuration.actual_sampling_rate * len(spike_times) / len_of_signal}, '
                                      f'abs error rate: {abs_error},'
                                      f'reconstruction error: {error_rate}, '
                                      f'number of spikes: {len(spike_times)} '
                                      f'threshold: {spiking_threshold}, ahp high:{ahp_high},'
                                      f'ahp period:{ahp_period}'
                                      f'sample len: {full_signal_len}')
                        if i % snapshot_interval == 0:
                            file_utils.write_array_to_csv(filename=stats_csv_file, data=reconstruction_stats)
                        if len(spike_times) > max_spike:
                            break
    file_utils.write_array_to_csv(filename=stats_csv_file, data=reconstruction_stats)

###################################################################################
############################ used for a reconstruction demo #######################
###################################################################################
if mode_for_running_this_driver == demo_mode:
    signal_from_wav_file = False
    sample_len = 10000
    sample_number = 5
    signal_norm_thrs = -1.0
    number_of_kernel = 10
    kernel_manager.init(number_of_kernel)
    full_signal = reconstruction_driver.get_signal(sample_number, read_from_wav=signal_from_wav_file)
    snippet, norm, _, _, _ = reconstruction_driver. \
        get_first_snippet_above_threshold_norm(full_signal, sample_len, signal_norm_thrs)
    actual_signal = snippet
    snippet = signal_utils.up_sample(snippet)
    snippet = snippet[: 40000]
    T = 1.0 / 44100 * 10
    N = 40000
    x = fftfreq(N, T)[:N // 2]
    y = fft(snippet)
    # plot_utils.plot_function(x, np.abs(y)[0:N // 2], title='fourier')

    signal_norm_square, signal_kernel_convolutions = reconstruction_manager.init_signal(snippet, configuration.mode,
                                                                                        parallel_mode=False)
    plot_utils.plot_function(snippet)
    selected_index = 5
    select_kernel_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    all_thresholds = [[] for i in range(number_of_kernel)]
    all_ys = []
    all_xs = []
    all_xs.append(list(range(len(snippet))))
    all_ys.append(snippet)
    locations = [-4000, 11500, 27000]
    # , 20000, 35000]
    krnl = kernel_manager.all_kernels[3]
    for l in locations:
        all_ys.append(krnl[::-1] / 1e4 - 0.004)
        all_xs.append(list(range(l, l + len(krnl))))
    plot_utils.plot_functions_in_one_plot(all_xs, all_ys, colors=['blue', 'red', 'red', 'red'])

    spike_times, spike_indexes, threshold_values, spikes_of_all_kernel = \
        spike_generator.calculate_spikes_for_all_kernels(signal_kernel_convolutions, select_kernel_indexes,
                                                         ahp_period=1000,
                                                         ahp_high=1e-5, threshold=4e-6, all_threshold=all_thresholds)
    recons_coeffs = reconstruction_manager.calculate_reconstruction(spike_times, spike_indexes, threshold_values)
    plot_utils.plot_functions_in_one_plot([signal_kernel_convolutions[selected_index],
                                           all_thresholds[selected_index]], colors=['blue', 'green'])

    import matplotlib.pyplot as plt
    import matplotlib.axes as ax
    import matplotlib.cm as cm

    # spikes_of_all_kernel = np.array(spikes_of_all_kernel)

    all_colors = []
    # [[] for i in range(number_of_kernel)]
    for i in range(len(recons_coeffs)):
        all_colors.append(recons_coeffs[i])
        # all_colors[spike_indexes[i]].append(recons_coeffs[i])
    # plt.eventplot(spikes_of_all_kernel, linelengths=0.3, linewidths=0.7, colors=all_colors)
    # plt.xlim(0, 22050)
    recons_signal = reconstruction_manager.get_reconstructed_signal(len(snippet),
                                                                    spike_times, spike_indexes, recons_coeffs)
    print(f'error rate is: {signal_utils.calculate_absolute_error_rate(snippet, recons_signal)}')
    plot_utils.plot_function(recons_signal)
    recons_coeffs = recons_coeffs - np.min(recons_coeffs) + 1
    recons_coeffs = np.log10(recons_coeffs)

    colors = cm.rainbow(np.linspace(0, 1, 100))
    all_colors = [
        colors[int(99 * (recons_coeffs[i] - np.min(recons_coeffs)) / (np.max(recons_coeffs) - np.min(recons_coeffs)))]
        for i
        in range(len(recons_coeffs))]
    plt.scatter(spike_times, spike_indexes, s=30, marker='|',
                c=all_colors)
    plt.xlim(0, 22050)
    # plt.plot(recons_coeffs)
    # ax.Axes.set_xscale(10, 'linear')
    # ticks = plt.get_xticks() / 441
    # plt.xticks(range(0, 1000, 100))
    plt.show()
