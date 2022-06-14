import configuration
import numpy as np
import reconstruction_manager
import reconstruction_driver
import plot_utils
import matplotlib.pyplot as plt
import kernel_manager
import signal_utils

################################################
##### This is created to do slim testings ######
################################################
ahp = configuration.ahp_period / 1.0
sample_number = 5
sample_len = 10000
number_of_kernel = 10
    # 2\
    # 10
select_kernel_indexes = [2, 3, 4, 5, 6, 7, 8, 9]
    # [0]\
    # [2, 3, 4, 5, 6, 7, 8, 9]
signal_norm_thrs = 1e-4
win_mode = False
win_size = 30
norm_thres = 1e-3
z_thrs = np.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3]) * 1e-8  # 1e-8
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
            window_size=win_size, norm_threshold=norm_thres,
            recompute_recons_coeff=True,
            show_z_vals=True, z_threshold=z_thrs, signal_norm_sq=norm * norm,
            selected_kernel_indexes=select_kernel_indexes, input_signal=this_snippet)
    print(f' fast error rate of reconstruction is: {error_rate_fast}')

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
