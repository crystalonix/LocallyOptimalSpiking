import numpy as np
import kernel_manager
import common_utils
import configuration
import time


def spike_and_reconstruct_iteratively(all_convolutions, window_mode=False, window_size=-1,
                                      norm_threshold_for_new_spike=0.1):
    p_matrix = []
    p_inv_matrix = []
    z_vals_now = np.zeros(len(all_convolutions))
    eta_vals_now = [None for i in range(len(all_convolutions))]
    beta_vals_now = [None for i in range(len(all_convolutions))]
    xip_vals_now = np.zeros(len(all_convolutions))
    z_vals_next = np.zeros(len(all_convolutions))
    xip_vals_next = np.zeros(len(all_convolutions))
    eta_vals_next = [None for i in range(len(all_convolutions))]
    beta_vals_next = [None for i in range(len(all_convolutions))]
    threshold_values = np.array([], dtype=float)
    recons_coeffs = np.array([], dtype=float)
    spike_times = np.array([], dtype=int)
    spike_indexes = np.array([], dtype=int)
    spike_counter = 0
    for t in range(len(all_convolutions[0]) - 1):
        z_max_this = -1
        this_spike_index = -1
        for i in range(len(all_convolutions)):
            if t + 1 < len(all_convolutions[i]):
                st = time.process_time()
                eta_next = calculate_eta(spike_times, spike_indexes, t + 1, i, window_mode, window_size)
                beta_next = calculate_beta(p_inv_matrix, eta_next)
                x_r_next = calculate_residual_ip(all_convolutions[i][t + 1], beta_next,
                                                 threshold_values, window_mode)
                z_next = x_r_next ** 2
                new_spike_norm = 1 - np.dot(eta_next, beta_next)
                z_next = z_next / new_spike_norm
                if configuration.debug:
                    print(f'time for eta and beta computation is: {time.process_time() - st}')
                #######################################################################
                ########### adds only one spike at a time on a local maxima ###########
                #######################################################################
                if new_spike_norm > norm_threshold_for_new_spike and t > 1 and z_next < z_vals_next[i] \
                        and z_vals_next[i] > z_vals_now[i] and z_max_this < z_vals_next[i]:
                    this_spike_index = i
                    this_threshold = all_convolutions[i][t]
                    z_max_this = z_vals_next[i]
                    z_prev_val = z_vals_now[i]
                    z_next_val = z_next
                    if z_max_this < 0 and configuration.debug:
                        print('this is a flaw')
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
            ##########################################################
            ######## TODO: if window mode compress this part #########
            spike_times = np.append(spike_times, t)
            spike_indexes = np.append(spike_indexes, this_spike_index)
            threshold_values = np.append(threshold_values, this_threshold)
            spike_counter = spike_counter + 1
            if configuration.verbose:
                print(f'produced spike number {spike_counter} at time {t}, by kernel {this_spike_index}, at threshold '
                      f'{this_threshold}, z_val {z_max_this}, z_prev {z_prev_val}, z_next {z_next_val}')
            # if configuration.verbose:
            #     print(f'generated a spike at time: {t} by the {this_spike_index}-th '
            #           f'kernel at threshold: {this_threshold}')
            ################################################################
            ##### Update the coefficients and p_matrix and the inverse #####
            ################################################################
            st = time.process_time()
            recons_coeffs = update_recons_coeffs(recons_coeffs, beta_vals_next[this_spike_index],
                                                 xip_vals_next[this_spike_index], z_vals_next[this_spike_index],
                                                 window_mode, window_size)
            if configuration.debug:
                print(f'time to calculate reconstruction coefficients: {time.process_time() - st}')
            st = time.process_time()
            p_matrix, p_inv_matrix = update_p_matrix_and_inv(p_matrix, eta_vals_next[this_spike_index],
                                                             window_mode, window_size)
            if configuration.debug:
                print(f'time to calculate p_inverse: {time.process_time() - st}')
    return spike_times, spike_indexes, threshold_values, recons_coeffs


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


def update_p_matrix_and_inv(p_matrix, eta_values, window_mode, window_size):
    """
    This method updates
    :rtype: return the updated p_matrix and its inverse
    :param p_matrix:
    :param eta_values:
    :param window_mode:
    """
    if len(p_matrix) == 0:
        p_new = np.array([[1.0]])
        return p_new, p_new.copy()

    assert p_matrix.shape[0] == len(eta_values)
    p_new = np.vstack((p_matrix, eta_values))
    p_new = np.hstack((p_new, np.append(eta_values, 1).reshape(-1, 1)))
    if window_mode:
        p_new = p_new[-min(window_size, len(eta_values) + 1):, -min(window_size, len(eta_values) + 1):]
    p_inv = common_utils.solve_for_inverse_by_torch(p_new)
    return p_new, p_inv
