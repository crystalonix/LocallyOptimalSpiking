import numpy as np
import common_utils
import configuration


def upsample(signal, up_factor=configuration.upsample_factor, interpolation_fn=common_utils.linear_interpolation):
    """
    upsamples a given signal by up_factor
    :param signal:
    :param up_factor:
    :param interpolation_fn: given two end points interpolates the signal in between
    """
    sig_len = signal.size
    up_signal = np.zeros(sig_len * up_factor + 1)

    for i in range(sig_len - 1):
        up_signal[i * up_factor: (i + 1) * up_factor] = interpolation_fn(signal[i], signal[i + 1], up_factor)

    up_signal[-1] = signal[-1]
    return up_signal


def calculate_convolution(f1, f2, samp_rate=configuration.sampling_rate):
    """
    Returns the convolution of two signals
    :param samp_rate:
    :param f1:
    :param f2:
    :return: result of the convolution
    """
    return np.convolve(f1, f2) / samp_rate


def get_signal_norm_square(signal, samp_rate=configuration.sampling_rate):
    """
    Returns the squared norm of signal
    :param signal:
    :param samp_rate:
    :return:
    """
    return np.dot(signal, signal) / samp_rate


def normalize_signals(all_signals):
    """
    This method normalizes a list of signals
    :param all_signals:
    :return:
    """
    normalized_signals = []
    for i in range(len(all_signals)):
        normalized_signals.append(normalize_signal(all_signals[i]))
    return normalized_signals


def normalize_signal(signal, samp_rate=configuration.sampling_rate):
    """
    This function returns the normalized signal
    :param signal:
    :param samp_rate:
    :return:
    """
    return signal / np.sqrt(get_signal_norm_square(signal, samp_rate))


def add_shifted_comp_signals(comp_signal, shifts, coefficients=None):
    length_of_full_signal = len(comp_signal) + int(np.max(shifts))
    components_matrix = np.zeros((length_of_full_signal, len(shifts)))
    for i in range(len(shifts)):
        this_shift = int(shifts[i])
        components_matrix[this_shift: len(comp_signal) + this_shift, i] = comp_signal
    if coefficients is None:
        coefficients = np.ones(len(shifts))
    print(f'check the shapes here: {components_matrix.shape} & {coefficients.shape}')
    return np.matmul(components_matrix, coefficients)


def zero_pad(this_snippet, zero_pad_len, both_sides):
    total_len = len(this_snippet) + zero_pad_len
    total_len = total_len + zero_pad_len if both_sides else total_len
    zero_pad_signal = np.zeros(total_len)
    zero_pad_signal[zero_pad_len:zero_pad_len+len(this_snippet)] = this_snippet
    return zero_pad_signal
