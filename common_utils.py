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
    print(T)
    return torch.lstsq(T, P)


def solve_for_coefficients_by_tf(P, T):
    P = tf.convert_to_tensor(P)
    T = tf.convert_to_tensor(T)
    return tf.linalg.lstsq(P, T, fast=True)
