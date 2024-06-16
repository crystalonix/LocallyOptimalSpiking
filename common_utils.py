#
# Copyright (c) 2024 Anik Chattopadhyay, Arunava Banerjee
#
# Author: Anik Chattopadhyay
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-nd/4.0/
#
# Note: This project is also subject to a provisional patent. The Creative Commons license
# applies to the documentation and code provided herein, but does not grant any rights to
# the patented invention.
#

import numpy as np
import torch
import tensorflow as tf

import configuration


def linear_interpolation(point1, point2, length):
    """
    linearly interpolates the data between two end points
    :param point1:
    :param point2:
    :param length:
    :return:
    """
    interpolated_sig = np.zeros(length)
    alpha = (point2 - point1) / length
    for i in range(length):
        interpolated_sig[i] = point1 + alpha * i
    return interpolated_sig


def quad_interpolation(point1, point2, length):
    """

    :param point1:
    :param point2:
    :param length:
    :return:
    """
    return NotImplemented


def matrixVectorProduct(matrix, vector):
    """
    Returns the multiplication of a matrix with a vector
    :param matrix:
    :param vector:
    :return:
    """
    return np.dot(matrix, vector)


def solve_linear_equn_by_conjuagte_gradient(p_matrix, threshold_crossing_values,
                                            thread_count=configuration.number_of_threads, number_of_steps=-1,
                                            SMALL_EPS=0.000001):
    """

    :param p_matrix:
    :param threshold_crossing_values:
    :param thread_count:
    :param number_of_steps:
    :param SMALL_EPS:
    :return:
    """
    counter = 0
    r = np.copy(threshold_crossing_values)
    x = np.zeros_like(threshold_crossing_values)
    p = r
    while (number_of_steps < 0) or (counter < number_of_steps):
        ap = matrixVectorProduct(p_matrix, p)
        r_old_prod = np.squeeze(np.dot(r.T, r))
        alpha = r_old_prod / np.squeeze(np.dot(p.T, ap))
        x = x + alpha * p
        r = r + -alpha * ap
        residual = max(abs(r))
        if residual < SMALL_EPS:
            break
        beta = np.dot(r.T, r) / r_old_prod
        p = r + beta * p
        counter = counter + 1
    return x


def solve_linear_equn_by_least_square(p_matrix, threshold_crossing_values):
    """
    This method solves a system of linear equations using least square method
    :param p_matrix:
    :param threshold_crossing_values:
    :return:
    """
    return np.linalg.lstsq(p_matrix, threshold_crossing_values)


def solve_for_coefficients(P, T, method='torch'):
    """

    :param P:
    :param T:
    :param method:
    :return:
    """
    if len(P) == 0:
        return torch.tensor([])
    if method == 'torch':
        return solve_for_coefficients_by_torch(P, T)
    elif method == 'tensorflow':
        return solve_for_coefficients_by_tf(P, T)
    elif method == 'numpy':
        return solve_linear_equn_by_least_square(P, T)
    return None


def solve_for_coefficients_by_torch(P, T):
    size = len(T)
    P = torch.tensor(P, dtype=torch.float64)
    T = torch.tensor(T, dtype=torch.float64)
    return torch.lstsq(T, P).solution


def solve_for_coefficients_by_tf(P, T):
    P = tf.convert_to_tensor(P)
    T = tf.convert_to_tensor(T)
    return tf.linalg.lstsq(P, T, fast=True)


def solve_for_inverse_by_torch(p_matrix):
    P = torch.tensor(p_matrix, dtype=torch.float64)
    return torch.inverse(P)


def p_inv_iteratively_torch(p_inv_old, new_col, new_col_ip, schur_complement=None, fixed=False):
    """
    :param fixed:
    :param schur_complement: this is the inverse of the norm of the perpendicular component
    of the new spike wrt to the span of existing kernels
    :param p_inv_old: (n-1)X(n-1) matrix which is the inverse of the P-matrix of the previous step
    :param new_col: the inner products of the new spike with other spikes as a column vector
    :param new_col_ip: p_inv_old X new_col
    """
    assert len(new_col) == len(p_inv_old)
    if schur_complement is None:
        schur_complement = 1 - np.dot(new_col, new_col_ip)
    p_inv_new = torch.zeros(len(new_col) + 1, len(new_col) + 1, dtype=torch.float64)

    torch.outer(torch.tensor(new_col_ip), torch.tensor(new_col_ip), out=p_inv_new[0:len(new_col), 0:len(new_col)])

    p_inv_new[len(new_col), 0: len(new_col)] = -1 * torch.tensor(new_col_ip.T)
    p_inv_new[0: len(new_col), len(new_col)] = -1 * torch.tensor(new_col_ip)
    p_inv_new[len(new_col), len(new_col)] = 1

    if fixed:
        p_inv_new = p_inv_new / schur_complement
    else:
        n = configuration.SCHUR_POWER
        p_inv_new = p_inv_new * (schur_complement ** n)
    # print(f'check the intermediate: \n {np.array(p_inv_new)}')
    p_inv_new[0:len(new_col), 0:len(new_col)] += p_inv_old
    # print(f'check the intermediate2: \n {np.array(p_inv_new)}')
    return p_inv_new


def sort_spikes_on_kernel_indexes(spike_times, spike_indexes, num_kernels, display=True):
    all_spikes = [None for i in range(num_kernels)]
    for i in range(len(spike_times)):
        ind = spike_indexes[i]
        if all_spikes[ind] is None:
            all_spikes[ind] = []
        all_spikes[ind].append(spike_times[i])
    if display:
        for i in range(num_kernels):
            print(f'spikes of kernel {i} are:\n {all_spikes[i]}')
    return all_spikes


# def matrix_sup_norm(matrix):


# sz = 2
# p_test = np.random.rand(sz, sz)
# p_test = p_test + p_test.T
# p_test = np.array([[1.0, 0.707], [0.707, 1.0]])
# # p_test[3:3] = 1.0
# eta = p_test[sz - 1, :sz - 1]
# print(f'check eta values: {eta}')
# p_test_inv_old = np.linalg.inv(p_test[:sz - 1, :sz - 1])
# p_test_inv_full = np.linalg.inv(p_test)
# beta = np.dot(p_test_inv_old, eta)

# schur_comp = 1.0 / (1.0 - np.dot(eta, beta))
# p_test_inv_new = p_inv_iteratively_torch(p_test_inv_old, eta, beta)
# print(f'check the new p_inv: {p_test_inv_full}')
# print(f'check the difference in two inverses:\n{p_test_inv_full - p_test_inv_new.numpy()}')


#
# print(f'check the Schur value: {schur_comp}')
# print(f'the original p: \n{p_test}')
# print(f'check the p full inv:\n {p_test_inv_full} \n full ans: {np.dot(p_test, p_test_inv_full)}')
# print(f'check the iterative full inv: \n{np.array(p_test_inv_new)}\n iter ans: {np.dot(p_test, p_test_inv_new)}')
def calculate_cond_number(p_matrix):
    """
    Returns the condition number of a given 2-D matrix
    :rtype: any 2-D matrix
    """
    return np.linalg.cond(p_matrix)


def multiply_by_transpose(c_matrix):
    """
    Reurns the product a matrix with its transpose
    :param c_matrix:
    :return: product of the matrix and its transpose
    """
    return np.dot(c_matrix, c_matrix.transpose())
