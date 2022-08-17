import configuration
import numpy as np
import reconstruction_manager
import reconstruction_driver
import plot_utils
import matplotlib.pyplot as plt
import kernel_manager
import signal_utils
import reconstruction_manager
import iterative_spike_generator

################################################
##### This is created to do slim testings ######
################################################
ahp = configuration.ahp_period / 1.0
sample_number = 666
sample_len = 10000
number_of_kernel = 10
signal_norm_thrs = 1e-4
win_mode = True

select_kernel_indexes = [i for i in range(number_of_kernel)]
win_size = 10
kernel_norm_thres = 1e-2
z_thrs = np.array([1e3 for i in range(number_of_kernel)]) * 1e-11  # 1e-8
# np.array([1e3, 1e3]) * 1e-8\

# z_thrs = np.array([0.0001, .001, .01, .1, 1.0]) * 1e-6

iterative_recons = True
direct_recons = True
rectify_coeffs = True
single_snippet = True
show_z = False
turn_on_recons_plot = True
show_z_scores_plot = True
club_z_score_with_proj_thrs = False

single_snippet_run = True
if single_snippet_run:
    kernel_manager.init(number_of_kernel)

    this_snippet, norm, _, _, _ = reconstruction_driver. \
        get_first_snippet_above_threshold_norm(sample_number, sample_len, signal_norm_thrs)

    # this_snippet = signal_utils.zero_pad(this_snippet, zero_pad_len=len(this_snippet), both_sides=True)
    this_snippet = signal_utils.upsample(this_snippet)
    # plot_utils.plot_function(this_snippet, title='padded signal')
    if iterative_recons:
        sp_times, sp_indexes, thrs_values, recons_coeffs, error_rate_fast, recons, \
        signal_kernel_convolutions, z_scores, kernel_projections, gamma_vals, recons_signal, gamma_vals_manual = \
            reconstruction_manager.drive_single_signal_reconstruction_iteratively(
                this_snippet, False, need_reconstructed_signal=True, window_mode=win_mode,
                window_size=win_size, norm_threshold=kernel_norm_thres,
                recompute_recons_coeff=True,
                show_z_vals=True, z_threshold=z_thrs, signal_norm_sq=norm * norm,
                selected_kernel_indexes=select_kernel_indexes, input_signal=this_snippet)
        plot_utils.plot_functions([this_snippet, recons], plot_titles=['original signal', 'reconstruction'])
        print(f' fast error rate of reconstruction is: {error_rate_fast} with {len(sp_times)} spikes'
              f', kernel norm threshold {kernel_norm_thres} and z-threshold {z_thrs[0]} for signal index: {sample_number}'
              f'window size: {win_size}')

    else:
        _, sp_times_1, sp_indexes_1, _, recons_coeffs_1, error_rate_fast_1, recons_1 = \
            reconstruction_manager.drive_select_snippet_reconstruction(sample_number, sample_len,
                                                                       norm_threshold=signal_norm_thrs,
                                                                       need_reconstructed_signal=True, ahp_period=ahp,
                                                                       selected_kernel_indexes=select_kernel_indexes)
    for this_index in select_kernel_indexes:
        plot_utils.plot_kernel_spike_profile(sp_times[sp_indexes == this_index], signal_kernel_convolutions[this_index],
                                             z_scores[this_index], kernel_projections[this_index],
                                             # other_fns=[gamma_vals[this_index],
                                             # gamma_vals[this_index] - signal_kernel_convolutions[this_index][
                                             #                                     :len(gamma_vals[this_index])],
                                             #            this_snippet, recons_signal,
                                             #            np.array(gamma_vals_manual[this_index]) - gamma_vals[this_index]
                                             #            # signal_kernel_convolutions[this_index][:len(gamma_vals[this_index])]
                                             #            ],
                                             # other_titles=['gamma vals', 'gamma minus conv',
                                             # 'input signal', 'reconstructed signal',
                                             #               'gamma minus gamma manual'],
                                             club_z_score_threshold=club_z_score_with_proj_thrs,
                                             kernel_index=this_index)
        plt.show()
else:
    ################### run signals over a grid of param values instead of a single run #######################
    sample_len = 20000
    number_of_kernel = 10
    signal_norm_thrs = 1e-4
    win_mode = True
    start = 10
    end = 100
    sample_numbers = np.array([i for i in range(start, end)])
    select_kernel_indexes = [i for i in range(number_of_kernel)]
    # [2, 3, 4, 5, 6, 7, 8, 9]
    win_sizes = [10]
    # [50, 30, 25, 20, 15, 10]
    kernel_norm_thresholds = np.array([1e-1])
    z_thrs_base = np.array([1e3 for i in range(number_of_kernel)])
    # np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])  # 1e-8
    z_thres_multiplier = np.array([1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17])
    # np.array([1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16, 1e-17])
    need_reconstructed_signal = False
    all_values_to_report = []
    ################# start the grid search here #####################
    kernel_manager.init(number_of_kernel)
    # plot_utils.plot_function(this_snippet, title='padded signal')
    if iterative_recons:
        for s in sample_numbers:
            this_snippet, norm, _, _, _ = reconstruction_driver. \
                get_first_snippet_above_threshold_norm(s, sample_len, signal_norm_thrs)
            this_snippet = signal_utils.upsample(this_snippet)
            signal_norm_square, signal_kernel_convolutions = \
                reconstruction_manager.init_signal(this_snippet, configuration.mode)
            for w in win_sizes:
                for k_t in kernel_norm_thresholds:
                    for z_mult in z_thres_multiplier:
                        sp_times, sp_indexes, thrs_values, recons_coeffs, z_scores, kernel_projections, \
                        gamma_vals, recons_signal, gamma_vals_manual = iterative_spike_generator. \
                            spike_and_reconstruct_iteratively(signal_kernel_convolutions, window_mode=win_mode,
                                                              window_size=w,
                                                              norm_threshold_for_new_spike=k_t,
                                                              z_thresholds=z_thrs_base * z_mult,
                                                              show_z_scores=True, signal_norm_square=norm * norm,
                                                              selected_kernel_indexes=select_kernel_indexes,
                                                              input_signal=this_snippet)
                        recons_coeffs = reconstruction_manager.calculate_reconstruction(sp_times, sp_indexes,
                                                                                        thrs_values)
                        recons = None
                        error_rate_fast = None
                        error_rate_fast = reconstruction_manager.calculate_reconstruction_error_rate_fast(recons_coeffs,
                                                                                                          thrs_values,
                                                                                                          signal_norm_square)
                        all_values_to_report.append(
                            [error_rate_fast, len(sp_times), k_t, z_thrs_base[0] * z_mult, s, w])
                        print(f' fast error rate of reconstruction is: {error_rate_fast} with {len(sp_times)} spikes'
                              f', kernel norm threshold {k_t} and z-threshold {z_thrs_base[0] * z_mult} '
                              f'for signal index: {s}'
                              f'window size: {w}')
                        if len(sp_times) > configuration.max_allowed_spikes:
                            break
            print(f'\n\n\n All report so far: {all_values_to_report}\n\n\n')

        # TODO: modify the else condition if run in non iterative mode
        else:
            _, sp_times_1, sp_indexes_1, _, recons_coeffs_1, error_rate_fast_1, recons_1 = \
                reconstruction_manager.drive_select_snippet_reconstruction(sample_number, sample_len,
                                                                           norm_threshold=signal_norm_thrs,
                                                                           need_reconstructed_signal=True,
                                                                           ahp_period=ahp,
                                                                           selected_kernel_indexes=select_kernel_indexes)
