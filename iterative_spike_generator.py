import numpy as np
import kernel_manager
import common_utils
import configuration
import time
import configuration
import plot_utils


def update_c_matrix(c_matrix, eta_values, zeta, beta_values, window_mode=False, window_size=10):
    """
    This method calculates the conditioner for the P-matrix which in turn is used in solving Pa = T efficiently
    :param c_matrix: c_matrix in the previous step
    :param eta_values: inner product of the new kernel with other kernels
    :param zeta: norm of the perpendicular component of the new kernel
    :param beta_values: coefficients of reconstruction of the new kernel in the span of existing kernels
    :param window_mode: whether windowed p-matrix
    :param window_size: size of the window
    :return: updated c_matrix
    """
    n = len(c_matrix)
    if n == 0:
        return np.ones((1, 1)), np.ones((1, 1))
    c_matrix_new = np.zeros((n + 1, n + 1))
    c_matrix_new[:n, :n] = c_matrix
    c_matrix_new[:-1, -1] = -beta_values / zeta
    c_matrix_new[-1, -1] = 1 / zeta
    p_inv = common_utils.multiply_by_transpose(c_matrix_new)
    return c_matrix_new, p_inv


def calculate_gamma_1(beta_next, threshold_values):
    if beta_next is None or len(beta_next) == 0:
        return 0
    return np.dot(beta_next, threshold_values)


def calculate_gamma_2(eta_next, recons_coeffs):
    if eta_next is None or len(eta_next) == 0:
        return 0
    return np.dot(eta_next, recons_coeffs)


def spike_and_reconstruct_iteratively(all_convolutions, window_mode=False, window_size=-1,
                                      norm_threshold_for_new_spike=0.1, z_thresholds=0.5, show_z_scores=False,
                                      signal_norm_square=None, selected_kernel_indexes=None,
                                      preconditioning=configuration.precondition_mode):
    p_matrix = []
    p_inv_matrix = []
    c_matrix = []
    z_vals_now = np.zeros(len(all_convolutions))
    eta_vals_now = [None for i in range(len(all_convolutions))]
    beta_vals_now = [None for i in range(len(all_convolutions))]
    xip_vals_now = np.zeros(len(all_convolutions))
    z_vals_next = np.zeros(len(all_convolutions))
    xip_vals_next = np.zeros(len(all_convolutions))
    eta_vals_next = [None for i in range(len(all_convolutions))]
    beta_vals_next = [None for i in range(len(all_convolutions))]
    z_scores = [None for i in range(len(all_convolutions))]
    gamma_vals = [None for i in range(len(all_convolutions))]
    kernel_projections = [None for i in range(len(all_convolutions))]
    threshold_values = np.array([], dtype=float)
    recons_coeffs = np.array([], dtype=float)
    spike_times = np.array([], dtype=int)
    spike_indexes = np.array([], dtype=int)

    # zeta_values will stores the perpendicular norm of the new spike kernels
    zeta_values = np.array([], dtype=float)
    spike_counter = 0
    residual_norm_square = signal_norm_square
    spike_counts = np.zeros(len(all_convolutions))
    for t in range(len(all_convolutions[0]) - 1):
        z_max_this = -1
        this_spike_index = -1
        for i in range(len(all_convolutions)):
            if selected_kernel_indexes is not None and i not in selected_kernel_indexes:
                continue
            if t + 1 < len(all_convolutions[i]):
                st = time.process_time()
                eta_next = calculate_eta(spike_times, spike_indexes, t + 1, i, window_mode, window_size)
                beta_next = calculate_beta(p_inv_matrix, eta_next)
                x_r_next = calculate_residual_ip(all_convolutions[i][t + 1], beta_next,
                                                 threshold_values, window_mode)
                if configuration.debug:
                    gamma = calculate_gamma_1(beta_next, threshold_values)
                    # gamma = calculate_gamma_2(eta_next, recons_coeffs)
                    if gamma_vals[i] is None:
                        gamma_vals[i] = [gamma]
                    else:
                        gamma_vals[i].append(gamma)
                z_next = x_r_next ** 2
                new_spike_norm_sq = 1 - np.dot(eta_next, beta_next)
                z_next = z_next / new_spike_norm_sq
                ###############################################################
                # TODO: currently z-score is normalized only in non-window mode
                # TODO: verify this carefully
                ###############################################################
                if configuration.z_score_by_residual_norm:
                    z_next = z_next / residual_norm_square
                if show_z_scores:
                    if z_scores[i] is None:
                        z_scores[i] = [z_next]
                        kernel_projections[i] = [new_spike_norm_sq]
                    else:
                        z_scores[i].append(z_next)
                        kernel_projections[i].append(new_spike_norm_sq)

                # TODO: uncomment this when the exploding condition number problem is solved

                # assert z_next >= 0
                if configuration.debug:
                    print(f'time for eta and beta computation is: {time.process_time() - st}')
                #######################################################################
                ########### adds only one spike at a time on a local maxima ###########
                #######################################################################
                zeta_val_at_t = kernel_projections[i][len(kernel_projections[i]) - 2]

                if zeta_val_at_t > norm_threshold_for_new_spike and t > 1 and z_next < z_vals_next[i] \
                        and z_vals_next[i] > z_vals_now[i] and z_max_this < z_vals_next[i]:
                    if window_mode or z_vals_next[i] > z_thresholds[i]:
                        this_spike_index = i
                        this_threshold = all_convolutions[i][t]
                        z_max_this = z_vals_next[i]
                        z_prev_val = z_vals_now[i]
                        z_next_val = z_next

                z_vals_now[i] = z_vals_next[i]
                eta_vals_now[i] = eta_vals_next[i]
                beta_vals_now[i] = beta_vals_next[i]
                xip_vals_now[i] = xip_vals_next[i]
                z_vals_next[i] = z_next
                eta_vals_next[i] = eta_next
                beta_vals_next[i] = beta_next
                xip_vals_next[i] = x_r_next

        ##########################################################
        ########### add the spike and spike index here ###########
        ##########################################################
        if this_spike_index > -1:
            print(f'producing spike #{len(spike_times) + 1}')
            ##########################################################
            ######## TODO: if window mode compress this part #########
            spike_times = np.append(spike_times, t)
            spike_indexes = np.append(spike_indexes, this_spike_index)
            threshold_values = np.append(threshold_values, this_threshold)
            zeta_values = np.append(zeta_values, np.sqrt(zeta_val_at_t))
            spike_counter = spike_counter + 1
            spike_counts[this_spike_index] += 1

            if configuration.verbose:
                if len(spike_times) % 2 == 0 and not preconditioning:
                    print(f'The condition number of the p-matrix is: {common_utils.calculate_cond_number(p_matrix)}'
                          f'and the length of p-matrix: {len(p_matrix)}')

            st = time.process_time()

            if configuration.debug:
                print(f'produced spike number {spike_counter} at time {t}, by kernel {this_spike_index}, at threshold '
                      f'{this_threshold}, z_val {z_max_this}, z_prev {z_prev_val}, z_next {z_next_val}')
                print(f'time to calculate reconstruction coefficients: {time.process_time() - st}')
            st = time.process_time()
            if preconditioning:
                # TODO: check if this has to be beta_next or beta_now && eta_vals now or next
                # TODO: remove the recomputation of p_matrix
                c_matrix, p_inv_matrix_new = update_c_matrix(c_matrix, eta_vals_now[this_spike_index],
                                                             np.sqrt(zeta_val_at_t), beta_vals_now[this_spike_index],
                                                             window_mode, window_size)
                # p_matrix, _ = update_p_matrix_and_inv(p_matrix, eta_vals_now[this_spike_index],
                #                                       beta_vals_now[this_spike_index], p_inv_matrix,
                #                                       window_mode, window_size)
                p_inv_matrix = p_inv_matrix_new
                recons_coeffs = np.dot(p_inv_matrix, threshold_values)
                residual_norm_square = signal_norm_square - np.dot(threshold_values, recons_coeffs)
                print(f'new residual norm square: {residual_norm_square}')
                # mat_1 = np.dot(c_matrix.transpose(), np.dot(p_matrix, c_matrix))
                # mat_2 = np.dot(p_matrix, np.dot(c_matrix, c_matrix.transpose()))
                # print(f'Spike number# {len(spike_times)} is produced with zeta score: {np.sqrt(zeta_val_at_t)}')
                # print(f'the norm of the mat_1-eye: {np.linalg.norm(mat_1 - np.eye(len(mat_1)))}'
                #       # f', {mat_1}'
                #       )
                # print(f'the norm of the mat_2-eye: {np.linalg.norm(mat_2 - np.eye(len(mat_2)))}'
                #       # f', {mat_2}'
                #       )
                print(f'norm of beta values is: {np.linalg.norm(beta_vals_now[this_spike_index])}')
            else:
                p_matrix, p_inv_matrix = update_p_matrix_and_inv(p_matrix, eta_vals_now[this_spike_index],
                                                                 beta_vals_now[this_spike_index], p_inv_matrix,
                                                                 window_mode, window_size)
                print(
                    f'checking the p product: {np.linalg.norm(np.dot(p_matrix, p_inv_matrix) - np.eye(len(p_matrix)))}')
                if window_mode:
                    recons_coeffs = update_recons_coeffs(recons_coeffs, beta_vals_now[this_spike_index],
                                                         xip_vals_now[this_spike_index], z_vals_now[this_spike_index],
                                                         window_mode, window_size)
                else:
                    recons_coeffs = np.dot(p_inv_matrix, threshold_values)
                residual_norm_square = signal_norm_square - np.dot(recons_coeffs, threshold_values)
                print(f'the residual square is: {residual_norm_square}')
                if configuration.debug:
                    print(f'time to calculate p_inverse: {time.process_time() - st}')
    print(f'spike counts: {spike_counts}')
    return spike_times, spike_indexes, threshold_values, recons_coeffs, z_scores, kernel_projections, gamma_vals


def calculate_beta(p_inv_matrix, eta):
    """
    This method returns the coefficients of projection of
    a kernel into the span of spike generating kernels
    :param p_inv_matrix: inv of the p-matrix formed by the previous spikes
    p.s. if we are operating in the window mode the appropriately sized p_inv_matrix must be passed
    :param eta: inner product of the kernel with rest of spike generating kernels
    """
    if len(eta) == 0:
        return []
    return np.dot(p_inv_matrix, eta)


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


def calculate_residual_ip(signal_kernel_ip, beta_vals, all_threshold_values, window_mode):
    """
    This function returns the inner product of the kernel with the residual signal
    :param signal_kernel_ip:
    :param beta_vals:
    :param all_threshold_values:
    :param window_mode: if the window mode is true the inner product is windowed residual ip
    :param window_size: size of the window
    :return:
    """
    if len(beta_vals) == 0:
        return signal_kernel_ip
    if window_mode:
        return signal_kernel_ip - np.dot(beta_vals, all_threshold_values[-len(beta_vals):])
    else:
        return signal_kernel_ip - np.dot(beta_vals, all_threshold_values)


def update_recons_coeffs(old_recons_coeffs, beta_values, kernel_residual_inner_prod, z_score, window_mode, window_size):
    """
    This method updates the coefficients of reconstruction after a new spike is generated
    :param old_recons_coeffs:
    :param beta_values:
    :param kernel_residual_inner_prod:
    :param z_score:
    :param window_mode:
    :param window_size:
    :return:
    """
    alpha_new = z_score / kernel_residual_inner_prod
    if len(old_recons_coeffs) == 0:
        return np.array([alpha_new])
    size = len(old_recons_coeffs)
    if window_mode:
        size = min(window_size, len(old_recons_coeffs))
    old_recons_coeffs[-size:] = old_recons_coeffs[-size:] - alpha_new * beta_values[-size:]
    return np.append(old_recons_coeffs, alpha_new)


def update_p_matrix_and_inv(p_matrix, eta_values, beta_values, p_inv_old,
                            window_mode=False, window_size=configuration.window_size,
                            direct_invert=configuration.direct_invert_p):
    """
    This method updates
    :param direct_invert:
    :param beta_values:
    :param window_size:
    :param p_inv_old:
    :rtype: return the updated p_matrix and its inverse
    :param p_matrix:
    :param eta_values:
    :param window_mode:
    """
    if len(p_matrix) == 0:
        p_new = np.array([[1.0]])
        return p_new, p_new.copy()
    assert p_matrix.shape[0] == len(eta_values)
    eta_values_new = eta_values.copy()
    last_element = 1.0
    p_new = np.vstack((p_matrix, eta_values_new))
    p_new = np.hstack((p_new, np.append(eta_values_new, last_element).reshape(-1, 1)))
    if window_mode:
        p_new = p_new[-min(window_size, len(eta_values) + 1):, -min(window_size, len(eta_values) + 1):]
        p_inv = common_utils.solve_for_inverse_by_torch(p_new)
    else:
        if direct_invert:
            p_inv = common_utils.solve_for_inverse_by_torch(p_new)
        else:
            p_inv = common_utils.p_inv_iteratively_torch(p_inv_old, eta_values, beta_values)
    return p_new, p_inv
