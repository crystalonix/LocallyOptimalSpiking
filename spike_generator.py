import concurrent.futures
import itertools
import time

import numpy as np

import configuration
import kernel_manager

delta_thres = 0
abs_thres = 0


def lateral_inhibition(time_diff, index1, index2, lateral_high_value=configuration.ahp_high_value,
                       lateral_refractory_period=configuration.ahp_period):
    exponent_value = pow(configuration.lateral_inhibition_exponent, index2 - index1)
    if time_diff > lateral_refractory_period:
        return 0
    else:
        return lateral_high_value * (1 - time_diff / lateral_refractory_period) / exponent_value


def calculate_spike_times_with_lateral_inhibition(all_convolutions, ahp_period=configuration.ahp_period,
                                                  ahp_high=configuration.ahp_high_value,
                                                  threshold=configuration.spiking_threshold,
                                                  selected_kernel_indexes=None, offset=0, spike_start_time=0,
                                                  end_time=-1):
    all_spike_times = []
    threshold_values = []
    spike_indexes = []
    spikes_of_each_kernel = [[] for i in range(len(all_convolutions))]

    # global abs_thres, delta_thres
    for i in range(spike_start_time - offset,
                   len(all_convolutions[selected_kernel_indexes[0]]) if end_time == -1 else end_time - offset):
        for index in range(len(all_convolutions)):
            if selected_kernel_indexes is not None and index not in selected_kernel_indexes:
                continue
            if i >= len(all_convolutions[index]):
                continue
            this_convolution = all_convolutions[index]
            last_spikes = spikes_of_each_kernel[index]
            ahp_effect_now = 0
            if configuration.single_ahp:
                time_diff = (i + offset) - last_spikes[len(last_spikes) - 1]
                if time_diff > ahp_period:
                    break
                else:
                    ahp_effect_now = ahp_effect_now + \
                                     ahp_high * ((ahp_period - time_diff) / ahp_period)
            else:
                for n in range(len(last_spikes) - 1, -1, -1):
                    time_diff = (i + offset) - last_spikes[n]
                    if time_diff > ahp_period:
                        break
                    else:
                        ahp_effect_now = ahp_effect_now + \
                                         ahp_high * ((ahp_period - time_diff) / ahp_period)

            threshold_now = threshold + ahp_effect_now
            # add the effect due to lateral inhibition
            for k in selected_kernel_indexes:
                if k != index and len(spikes_of_each_kernel[k]) != 0 and \
                        (index + configuration.lateral_neighborhood) >= k >= (
                        index - configuration.lateral_neighborhood):
                    time_diff = spikes_of_each_kernel[k][len(spikes_of_each_kernel[k]) - 1]
                    threshold_now = threshold_now + lateral_inhibition(time_diff, index, k,
                                                                       lateral_high_value=configuration.ahp_high_value,
                                                                       lateral_refractory_period=configuration.ahp_period)
            if threshold_now <= this_convolution[i]:
                if configuration.debug:
                    print(f' threshold: {threshold_now} and convolution: {this_convolution[i]} and convolution prior:'
                          f'{-1 if i <= 0 else this_convolution[i - 1]} at time {i} with last spike '
                          f'{-1 if len(spikes_of_each_kernel[index]) == 0 else spikes_of_each_kernel[index][-1]}'
                          f' kernel index:{index}')
                spikes_of_each_kernel[index].append(i + offset)
                all_spike_times.append(i + offset)
                spike_indexes.append(index)
                # if spike_counts is not None:
                #     spike_counts[index] = spike_counts[index] + 1
                if configuration.quantized_threshold_transmission:
                    threshold_values.append(threshold_now)
                else:
                    threshold_values.append(this_convolution[i])
                # if configuration.quantized_threshold_transmission:
                #     abs_thres = abs_thres + abs(this_convolution[i])
                #     delta_thres = delta_thres + abs(this_convolution[i] - threshold_now)
    return all_spike_times, spike_indexes, threshold_values


def calculate_spike_times(all_convolutions, ahp_period=configuration.ahp_period, ahp_high=configuration.ahp_high_value,
                          threshold=configuration.spiking_threshold,
                          selected_kernel_indexes=None, offset=0, spike_start_time=0,
                          end_time=-1, each_kernel_spikes=None):
    """
    Computes the spikes for all kernels
    :param end_time:
    :param all_convolutions:
    :param ahp_period:
    :param ahp_high:
    :param threshold:
    :param selected_kernel_indexes:
    :param offset:
    :param spike_start_time:
    :param each_kernel_spikes:
    :return:
    """
    global abs_thres, delta_thres
    delta_thres = 0
    abs_thres = 0
    if configuration.verbose:
        start_time = time.process_time()
    # spike_indexes = []
    # spike_times = []
    # threshold_values = []
    if configuration.verbose:
        spike_counts = np.zeros(len(all_convolutions))
    else:
        spike_counts = None
    # TODO: depreciated scheme of calculating spikes individually for each kernel
    # for i in range(len(all_convolutions)):
    #     if selected_kernel_indexes is not None and i not in selected_kernel_indexes:
    #         continue
    #     this_kernel_spikes, ths = calculate_spikes_for_one_kernel(all_convolutions, i, ahp_period, ahp_high, threshold)
    #     spike_times = np.concatenate([spike_times, this_kernel_spikes])
    #     spike_indexes = np.concatenate([spike_indexes, int(i) * np.ones(len(this_kernel_spikes))]).astype(int)
    #     if configuration.verbose:
    #         spike_counts[i] = len(this_kernel_spikes)
    #     threshold_values = np.concatenate([threshold_values, ths])
    if configuration.parallel_convolution:
        spike_times, spike_indexes, threshold_values = \
            calculate_spikes_for_all_kernels_parallel(all_convolutions, selected_kernel_indexes, ahp_period,
                                                      ahp_high, threshold, spike_counts,
                                                      offset=offset,
                                                      start_time=spike_start_time,
                                                      end_time=end_time,
                                                      each_kernel_spikes=each_kernel_spikes)
    else:
        spike_times, spike_indexes, threshold_values, _ = \
            calculate_spikes_for_all_kernels(all_convolutions, selected_kernel_indexes, ahp_period,
                                             ahp_high, threshold, spike_counts,
                                             offset=offset,
                                             start_time=spike_start_time,
                                             end_time=end_time,
                                             each_kernel_spikes=each_kernel_spikes)
    if len(spike_times) == 0:
        return [], [], []
    if configuration.quantized_threshold_transmission:
        print(f'check the spike counts here: {len(spike_times)}')
        print(f' the percentage error is transmitting threshold: {100 * delta_thres / abs_thres}')
    if configuration.verbose:
        print(f'check the spike counts here: {spike_counts}')
        print(f'number of spikes: {len(spike_times)} and '
              f'time to compute the spike: {time.process_time() - start_time}s')
    return spike_times, spike_indexes, threshold_values


def calculate_spikes_for_one_kernel(all_convolutions, kernel_index, ahp_period=configuration.ahp_period,
                                    ahp_high=configuration.ahp_high_value,
                                    threshold=configuration.spiking_threshold):
    """
    This method computes spikes generated by a given kernel
    :param ahp_high:
    :param all_convolutions:
    :param ahp_period:
    :param threshold:
    :param kernel_index:
    :return:
    """
    last_spike_time = -1
    this_kernel_spike_times = []
    threshold_values = []
    this_convolution = all_convolutions[kernel_index]
    last_spikes = []
    global abs_thres, delta_thres
    for i in range(len(this_convolution)):
        ahp_effect_now = 0
        if last_spike_time > -1:
            for n in range(len(last_spikes) - 1, -1, -1):
                time_diff = i - last_spikes[n]
                if time_diff > ahp_period:
                    break
                else:
                    ahp_effect_now = ahp_effect_now + \
                                     ahp_high * ((ahp_period - time_diff) / ahp_period)
        threshold_now = threshold + ahp_effect_now
        if threshold_now <= this_convolution[i]:
            # print(f' threshold: {threshold_now} and convolution: {this_convolution[i]} and convolution prior:'
            #       f'{-1 if i <= 0 else this_convolution[i - 1]} at time {i} with last spike {last_spike_time}')
            last_spike_time = i
            last_spikes.append(i)
            this_kernel_spike_times.append(i)
            if configuration.quantized_threshold_transmission:
                threshold_values.append(threshold_now)
            else:
                threshold_values.append(this_convolution[i])
            if configuration.debug:
                abs_thres = abs_thres + abs(this_convolution[i])
                delta_thres = delta_thres + abs(threshold_now - this_convolution[i])
                print(f'percentage error for this threshold: '
                      f'{100 * abs(threshold_now - this_convolution[i]) / abs(this_convolution[i])} '
                      f' index: {kernel_index} and time: {i}\n'
                      f' and the total percentage is: {100 * delta_thres / abs_thres}')
            # print(f' threshold: {threshold_now} and convolution: {this_convolution[i]} and convolution prior:'
            #       f'{-1 if i <= 0 else this_convolution[i - 1]}')
    return this_kernel_spike_times, threshold_values


def init():
    global abs_thres, delta_thres
    abs_thres = 0
    delta_thres = 0


def get_threshold_transmision_error_rate():
    return delta_thres / abs_thres


def calculate_spikes_for_all_kernels(all_convolutions, selected_kernel_indexes, ahp_period=configuration.ahp_period,
                                     ahp_high=configuration.ahp_high_value, threshold=configuration.spiking_threshold,
                                     spike_counts=None, offset=0, start_time=0, end_time=-1, each_kernel_spikes=None,
                                     all_threshold=None):
    """
    This method computes spikes generated by all kernels
    :param all_threshold:
    :param end_time:
    :param start_time:
    :param each_kernel_spikes:
    :param offset:
    :param ahp_high:
    :param spike_counts: counts of individual kernel spikes
    :param selected_kernel_indexes:
    :param all_convolutions:
    :param ahp_period:
    :param threshold: fill this array of threshold values of each kernel at all times if supplied
    :return:
    """
    all_spike_times = []
    threshold_values = []
    spike_indexes = []
    if each_kernel_spikes is None:
        spikes_of_each_kernel = [[] for i in range(len(all_convolutions))]
    else:
        spikes_of_each_kernel = each_kernel_spikes
    global abs_thres, delta_thres
    for i in range(start_time - offset,
                   len(all_convolutions[selected_kernel_indexes[0]]) if end_time == -1 else end_time - offset):
        for index in range(len(all_convolutions)):
            if selected_kernel_indexes is not None and index not in selected_kernel_indexes:
                continue
            if i >= len(all_convolutions[index]):
                continue
            this_convolution = all_convolutions[index]
            last_spikes = spikes_of_each_kernel[index]
            ahp_effect_now = 0
            for n in range(len(last_spikes) - 1, -1, -1):
                time_diff = (i + offset) - last_spikes[n]
                if time_diff > ahp_period:
                    break
                else:
                    ahp_effect_now = ahp_effect_now + \
                                     ahp_high * ((ahp_period - time_diff) / ahp_period)

            threshold_now = threshold + ahp_effect_now
            if all_threshold is not None:
                all_threshold[index].append(threshold_now)
            if threshold_now <= this_convolution[i]:
                if configuration.debug:
                    print(f' threshold: {threshold_now} and convolution: {this_convolution[i]} and convolution prior:'
                          f'{-1 if i <= 0 else this_convolution[i - 1]} at time {i} with last spike '
                          f'{-1 if len(spikes_of_each_kernel[index]) == 0 else spikes_of_each_kernel[index][-1]}'
                          f' kernel index:{index}')
                spikes_of_each_kernel[index].append(i + offset)
                all_spike_times.append(i + offset)
                spike_indexes.append(index)
                if spike_counts is not None:
                    spike_counts[index] = spike_counts[index] + 1
                if configuration.quantized_threshold_transmission:
                    threshold_values.append(threshold_now)
                else:
                    threshold_values.append(this_convolution[i])
                if configuration.quantized_threshold_transmission:
                    abs_thres = abs_thres + abs(this_convolution[i])
                    delta_thres = delta_thres + abs(this_convolution[i] - threshold_now)
    return all_spike_times, spike_indexes, threshold_values, spikes_of_each_kernel


def calculate_spikes_for_all_kernels_parallel(all_convolutions, selected_kernel_indexes,
                                              ahp_period=configuration.ahp_period,
                                              ahp_high=configuration.ahp_high_value,
                                              threshold=configuration.spiking_threshold,
                                              spike_counts=None, offset=0, start_time=0, end_time=-1,
                                              each_kernel_spikes=None,
                                              all_threshold=None):
    """
    This method computes spikes generated by all kernels in parallel
    :param all_threshold:
    :param end_time:
    :param start_time:
    :param each_kernel_spikes:
    :param offset:
    :param ahp_high:
    :param spike_counts: counts of individual kernel spikes
    :param selected_kernel_indexes:
    :param all_convolutions:
    :param ahp_period:
    :param threshold: fill this array of threshold values of each kernel at all times if supplied
    :return:
    """
    if each_kernel_spikes is None:
        spikes_of_each_kernel = [[] for i in range(len(all_convolutions))]
    else:
        spikes_of_each_kernel = each_kernel_spikes
    results = [None for i in range(kernel_manager.num_kernels)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=configuration.number_of_threads) as executor:
        for i, result in enumerate(executor.map(single_kernel_spike_gen, [i for i in range(kernel_manager.num_kernels)],
                                                itertools.repeat(start_time), itertools.repeat(end_time),
                                                itertools.repeat(offset), all_convolutions,
                                                spikes_of_each_kernel, itertools.repeat(threshold),
                                                itertools.repeat(ahp_period), itertools.repeat(ahp_high),
                                                itertools.repeat(selected_kernel_indexes),
                                                itertools.repeat(all_threshold), itertools.repeat(spike_counts)
                                                )):
            spikes_of_each_kernel[i] = spikes_of_each_kernel[i] + result[0]
            results[i] = list(zip(result[0], result[1], result[2]))
    spikes_and_thresholds = []
    for i in range(len(results)):
        spikes_and_thresholds = spikes_and_thresholds + results[i]
    if len(spikes_and_thresholds) == 0:
        return [], [], []
    spikes_and_thresholds = sorted(spikes_and_thresholds, key=lambda x: x[0])
    spikes_and_thresholds = list(zip(*spikes_and_thresholds))
    return list(spikes_and_thresholds[0]), list(spikes_and_thresholds[1]), list(spikes_and_thresholds[2])


def single_kernel_spike_gen(kernel_index, start_time, end_time, offset, kernel_conv, last_spikes_of_this_kernel,
                            threshold=configuration.spiking_threshold, ahp_period=configuration.ahp_period,
                            ahp_high=configuration.ahp_high_value, selected_kernel_indexes=None,
                            all_threshold=None, spike_counts=None):
    """
    This method is for generating all spikes from a single kernel to be in parallel processing mode
    :param last_spikes_of_this_kernel:
    :param kernel_index: index of the kernel
    :param start_time: start time for spike calculation
    :param end_time: end time for spike calculation
    :param offset: time offset of the first convolution index
    :param kernel_conv: convolution of the kernel with signal
    :param threshold: spiking threshold
    :param ahp_period: refractory period
    :param ahp_high: elevated threshold after ahp
    :param selected_kernel_indexes: set of filtered kernels in use for spike generation
    :param all_threshold:
    :param spike_counts: count of spikes by each kernel
    :return:
    """
    spike_times = []
    spike_indexes = []
    spike_thresholds = []
    if selected_kernel_indexes is not None and kernel_index not in selected_kernel_indexes:
        return spike_times, spike_indexes, spike_thresholds
    # deep copy of last spikes
    last_spikes = list(last_spikes_of_this_kernel)
    for i in range(start_time - offset,
                   len(kernel_conv) if end_time == -1 else end_time - offset):
        ahp_effect_now = 0
        if configuration.single_ahp:
            time_diff = (i + offset) - last_spikes[len(last_spikes) - 1]
            if time_diff > ahp_period:
                break
            else:
                ahp_effect_now = ahp_effect_now + \
                                 ahp_high * ((ahp_period - time_diff) / ahp_period)
        else:
            for n in range(len(last_spikes) - 1, -1, -1):
                time_diff = (i + offset) - last_spikes[n]
                if time_diff > ahp_period:
                    break
                else:
                    ahp_effect_now = ahp_effect_now + \
                                     ahp_high * ((ahp_period - time_diff) / ahp_period)

        threshold_now = threshold + ahp_effect_now
        if all_threshold is not None:
            all_threshold[kernel_index].append(threshold_now)
        if threshold_now <= kernel_conv[i]:
            if configuration.debug:
                print(f' threshold: {threshold_now} and convolution: {kernel_conv[i]} and convolution prior:'
                      f'{-1 if i <= 0 else kernel_conv[i - 1]} at time {i} with last spike '
                      f'{-1 if len(last_spikes) == 0 else last_spikes[-1]}'
                      f' kernel index:{kernel_index}')
            last_spikes.append(i + offset)
            spike_times.append(i + offset)
            spike_indexes.append(kernel_index)
            if spike_counts is not None:
                spike_counts[kernel_index] = spike_counts[kernel_index] + 1
            if configuration.quantized_threshold_transmission:
                spike_thresholds.append(threshold_now)
            else:
                spike_thresholds.append(kernel_conv[i])
            if configuration.quantized_threshold_transmission:
                raise NotImplementedError('quantized threshold transmission not implemented in parallel mode')
    return spike_times, spike_indexes, spike_thresholds
