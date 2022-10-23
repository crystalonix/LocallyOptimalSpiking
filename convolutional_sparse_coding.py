import signal_utils
import configuration
import numpy as np
import time
import reconstruction_driver
import math
import kernel_manager
import file_utils


def omp_on_signal(signal, max_spike=configuration.max_spike_count,
                  select_kernel_indexes=None, sampling_rate=configuration.sampling_rate, filters=None, samp_number=-1):
    error_val_omp_new = signal_utils.get_signal_norm_square(signal)
    convs = calculate_signal_kernel_convs_with_zero_pad(signal, select_kernel_indexes)
    print('initial error value is: ', error_val_omp_new)
    eps = 1e-4 * error_val_omp_new
    max_itr = max_spike
    i = 0
    errors_omp_new = np.array([error_val_omp_new])
    initial_time = time.time()
    while i < max_itr and error_val_omp_new > eps:
        # find the max inner product of all
        max_result = np.where(convs == np.amax(convs))
        min_result = np.where(convs == np.amin(convs))
        max_index = (max_result[0][0], max_result[1][0])
        min_index = (min_result[0][0], min_result[1][0])
        # find out the maximum inner product here
        if convs[max_index[0]][max_index[1]] > np.abs(convs[min_index[0]][min_index[1]]):
            real_max_index = max_index
            real_max_value = convs[max_index[0]][max_index[1]]
        else:
            real_max_index = min_index
            real_max_value = convs[min_index[0]][min_index[1]]

        # update the signal snippet and all the convolution values
        i = i + 1

        # choose the atom here
        atom = np.zeros(len(signal))
        # flip the filter here because the kernels are casual
        kernel_len = np.min([len(filters[select_kernel_indexes[real_max_index[0]]]), real_max_index[1]])
        atom[real_max_index[1]:max(0, real_max_index[1] - kernel_len):-1] = \
            filters[select_kernel_indexes[real_max_index[0]]][:min(kernel_len, real_max_index[1])]
        if i == 1:
            atom = np.sqrt(sampling_rate) * atom / np.linalg.norm(atom)
            all_atoms = np.array([atom])
        else:
            before = time.time()
            atom = gram_schmidt(all_atoms.T, atom)
            after = time.time()
            all_atoms = np.append(all_atoms, [atom], axis=0)
        ip = np.inner(signal, atom) / sampling_rate
        signal = signal - ip * atom
        convs = calculate_signal_kernel_convs_with_zero_pad(signal, selected_kernel_indexes=select_kernel_indexes)
        error_val_omp_new = signal_utils.get_signal_norm_square(signal)
        errors_omp_new = np.append(errors_omp_new, error_val_omp_new)
        err_rate = error_val_omp_new / errors_omp_new[0]
        print(f'error rate at step {i} is: {err_rate}')
        if configuration.compute_time and configuration.debug:
            if i % 10 == 0:
                print('time taken in next 10 steps', time.time() - initial_time)
        if i % sampling_step_len == 0:
            reconstruction_stats.append([samp_number, err_rate, i/snippet_length, time.time()-initial_time])
            file_utils.write_array_to_csv(filename=reports_csv, data=reconstruction_stats)
    print(errors_omp_new)


def gram_schmidt(W, v, sampling_rate=configuration.sampling_rate):
    """
    Return an orthonormal kernel wrt an existing set of orthonormal kernels
    :param v: input kernel
    :param W: orthonormal set of kernels
    :type sampling_rate: rate at which the signal is sampled, is used to calculate the norm
    """
    v = v - np.matmul(W, np.matmul(W.T, v) / sampling_rate)
    norm = np.linalg.norm(v) / np.sqrt(sampling_rate)
    if norm == 0:
        return np.zeros(len(v))
    else:
        return v / norm


def calculate_signal_kernel_convs_with_zero_pad(snippet, selected_kernel_indexes=None):
    """
    This function calculates the convolution of a signal with the given kernels
    :param selected_kernel_indexes:
    :param snippet:
    :return:
    """
    all_convolutions = np.array([
        signal_utils.calculate_convolution(kernel_manager.all_kernels[selected_kernel_indexes[0]], snippet)])
    for i in range(1, len(selected_kernel_indexes)):
        conv = np.zeros(len(all_convolutions[0]))
        c = signal_utils.calculate_convolution(kernel_manager.all_kernels[selected_kernel_indexes[i]], snippet)
        conv[:len(c)] = c
        all_convolutions = np.append(all_convolutions, [conv], axis=0)
    return all_convolutions


sample_numbers = [i for i in range(1, 20)]
up_factor = 10
snippet_length = 5000
len_with_zero_pad = 90000
initial_zero_pad_len = 0
signal_from_wav_file = False
offset = 0
sampling_step_len = 50
reconstruction_stats = []
max_spike_count = 1500
for sample_number in sample_numbers:
    signal = reconstruction_driver.get_signal(sample_number, read_from_wav=signal_from_wav_file)
    signal = signal[offset:offset + snippet_length]
    signal = signal_utils.up_sample(signal, up_factor=up_factor)

    signal_snippet = np.zeros(len_with_zero_pad)
    signal_snippet[initial_zero_pad_len:initial_zero_pad_len + (len(signal) - offset)] = signal[offset:]

    number_of_kernel = 10
    # exclude some of the very low frequency kernel to make it computationally efficient
    select_kernel_indexes = [i for i in range(math.ceil(number_of_kernel / 10), number_of_kernel)]

    i = 0

    kernel_manager.init(number_of_kernels=number_of_kernel)
    fltrs = kernel_manager.all_kernels
    reports_csv = 'sparse_code.csv'
    omp_on_signal(signal_snippet, select_kernel_indexes=select_kernel_indexes,
                  max_spike=max_spike_count,filters=fltrs, samp_number=sample_number)
