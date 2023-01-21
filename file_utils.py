import matplotlib.pyplot as plt
import numpy as np
import csv
import plot_utils
import seaborn as sns
import pandas as pd


def read_1D_np_array(filename, delim=','):
    """
    This function reads values separated by delim from a file into a 1D array
    :param filename:
    :param delim:
    :return:
    """
    return np.loadtxt(filename, delimiter=delim)


def write_1D_np_array(filename, data, delim=','):
    """
    This method writes a numpy array into a text file
    :param filename:
    :param data:
    :param delim:
    """
    np.savetxt(filename, data, delimiter=delim)


def write_array_to_csv(filename, data, delim=','):
    """
    This method writes an array into a csv file with a specified delimiter
    :param filename:
    :param data:
    :param delim:
    """
    with open(filename, "w") as f:
        writer = csv.writer(f, delimiter=delim)
        writer.writerows(data)


def read_numpy_array_from_csv(filename, delim=',', data_type=float):
    """
    This method reads a numpy array from a csv file
    :param data_type:
    :param dtype:
    :param filename:
    :param delim:
    :return:
    """
    with open(filename, newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=delim))
    return np.array(data, dtype=data_type)


###########################################################################################
# following piece of code is used to generate error scatter plot to be used in the paper #
###########################################################################################
# reports_csv = '../csvresults/recons_reports_on_40_kernel_master.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# threshold_spike_rate = 0.02
# results = read_numpy_array_from_csv(reports_csv)
# filtered_results = results[:, [1, 4]]
# # filter those with early termination due to processing error, error rate has to lie between 0 and 1
# filtered_results = filtered_results[0 <= filtered_results[:, 0]]
# filtered_results = filtered_results[filtered_results[:, 0] < 1]
# filtered_results[:, 0] = -10 * np.log10(filtered_results[:, 0])
# # filtered_results[:, 1] = filtered_results[:, 1] / 10
# # set a threshold for spike rate
# filtered_results = filtered_results[threshold_spike_rate <= filtered_results[:, 1]]
# df = pd.DataFrame(filtered_results, columns=['SNR in DB', 'spike rate as a fraction of Nyquist rate = 44.1kHz'])
# sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz', data=df, scatter_kws={"s": 3},
#            line_kws={'color': 'purple'})
# print('done')

###########################################################################################
#   following piece of code is used to plot of comparative study between our technique    #
#   and convolutional sparse code techniques applied on a relatively small dataset        #
###########################################################################################

# reports_csv = '../csvresults/master_small_report.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# threshold_spike_rate = 0.02
# max_spike_threshold_rate = 0.3
# small_set_results = read_numpy_array_from_csv(reports_csv)
# smallset_filtered_results = small_set_results[:, [1, 4]]
# smallset_filtered_results = smallset_filtered_results[0 <= smallset_filtered_results[:, 0]]
# smallset_filtered_results = smallset_filtered_results[smallset_filtered_results[:, 0] < 1]
# smallset_filtered_results[:, 0] = -10 * np.log10(smallset_filtered_results[:, 0])
# smallset_filtered_results[:, 1] = smallset_filtered_results[:, 1] / 10
# # set a threshold for spike rate
# smallset_filtered_results = smallset_filtered_results[threshold_spike_rate <= smallset_filtered_results[:, 1]]
# smallset_filtered_results = smallset_filtered_results[max_spike_threshold_rate >= smallset_filtered_results[:, 1]]
# df1 = pd.DataFrame(smallset_filtered_results,
#                    columns=['SNR in DB', 'spike rate as a fraction of Nyquist rate = 44.1kHz'])
# # sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz', data=df1, scatter_kws={"s": 3},
# #            line_kws={'color': 'purple'})
#
# reports_csv = '../csvresults/master_sparse_code.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# threshold_spike_rate = 0.02
# sparse_code_results = read_numpy_array_from_csv(reports_csv)
# filtered_sparse_code_results = sparse_code_results[:, [1, 2]]
# filtered_sparse_code_results = filtered_sparse_code_results[0 <= filtered_sparse_code_results[:, 0]]
# filtered_sparse_code_results = filtered_sparse_code_results[filtered_sparse_code_results[:, 0] < 1]
# filtered_sparse_code_results[:, 0] = -10 * np.log10(filtered_sparse_code_results[:, 0])
# # set a threshold for spike rate
# filtered_sparse_code_results = filtered_sparse_code_results[threshold_spike_rate <= filtered_sparse_code_results[:, 1]]
# df2 = pd.DataFrame(filtered_sparse_code_results,
#                    columns=['SNR in DB', 'spike rate as a fraction of Nyquist rate = 44.1kHz'])
#
# concatenated_df = pd.concat([df1.assign(dataset='Our framework'),
#                              df2.assign(dataset='COMP')])
# sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz',
#            data=concatenated_df, hue='dataset', scatter_kws={"s": 3})
# print('done')

######################################################################################################
#           following piece of code is used to show a plot comparing the runtimes of our             #
#   technique  vs convolutional sparse code techniques applied on a relatively small dataset         #
######################################################################################################
#
# reports_csv = '../csvresults/recons_reports_for_time.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# threshold_spike_rate = 0.0
# max_spike_threshold_rate = 0.6
# # TODO: set this properly
# time_col_index = 8
# small_set_results_with_time = read_numpy_array_from_csv(reports_csv)
# smallset_filtered_results_with_time = small_set_results_with_time[:, [1, 4, time_col_index]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[0 <= smallset_filtered_results_with_time[:, 0]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[smallset_filtered_results_with_time[:, 0] < 1]
# smallset_filtered_results_with_time[:, 0] = -10 * np.log10(smallset_filtered_results_with_time[:, 0])
# smallset_filtered_results_with_time[:, 1] = smallset_filtered_results_with_time[:, 1] / 10
# # set a threshold for spike rate
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[threshold_spike_rate <= smallset_filtered_results_with_time[:, 1]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[max_spike_threshold_rate >= smallset_filtered_results_with_time[:, 1]]
# smallset_filtered_results_with_time = smallset_filtered_results_with_time[:, [1, 2]]
# df1 = pd.DataFrame(smallset_filtered_results_with_time,
#                    columns=['spike rate as a fraction of Nyquist rate = 44.1kHz', 'Time in second'])
# # sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz', data=df1, scatter_kws={"s": 3},
# #            line_kws={'color': 'purple'})
#
# reports_csv = '../csvresults/master_sparse_code.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# threshold_spike_rate = 0.02
# time_col_index = 3
# sparse_code_results_with_time = read_numpy_array_from_csv(reports_csv)
# filtered_sparse_code_results_with_time = sparse_code_results_with_time[:, [1, 2, time_col_index]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[0 <= filtered_sparse_code_results_with_time[:, 0]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[filtered_sparse_code_results_with_time[:, 0] < 1]
# filtered_sparse_code_results_with_time[:, 0] = -10 * np.log10(filtered_sparse_code_results_with_time[:, 0])
# # set a threshold for spike rate
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[threshold_spike_rate <= filtered_sparse_code_results_with_time[:, 1]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[max_spike_threshold_rate >= filtered_sparse_code_results_with_time[:, 1]]
#
# filtered_sparse_code_results_with_time = filtered_sparse_code_results_with_time[:, [1, 2]]
# df2 = pd.DataFrame(filtered_sparse_code_results_with_time,
#                    columns=['spike rate as a fraction of Nyquist rate = 44.1kHz', 'Time in second'])
#
# concatenated_df = df2
#     # pd.concat([df1.assign(dataset='Our framework'),
#     #                          df2.assign(dataset='COMP')])
# sns.lmplot(x='Time in second', y='spike rate as a fraction of Nyquist rate = 44.1kHz',
#            data=concatenated_df, scatter_kws={"s": 3})
#
# print('done')

###########################################################################################
#   following piece of code is used to plot the results of the comparative study between  #
#     our technique and advanced sporco based convolutional sparse code techniques        #
###########################################################################################

# reports_csv = '../csvresults/recons_reports_csc_comparison_10_kernel_lime.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# length_col_index = 10
# threshold_spike_rate = 0.02
# max_spike_threshold_rate = .6
# min_signal_length = 20000
# max_signal_length = 60000
# small_set_results = read_numpy_array_from_csv(reports_csv)
# smallset_filtered_results = small_set_results[:, [1, 4, length_col_index]]
# smallset_filtered_results = smallset_filtered_results[0 <= smallset_filtered_results[:, 0]]
# smallset_filtered_results = smallset_filtered_results[smallset_filtered_results[:, 0] < 1]
# smallset_filtered_results[:, 0] = -10 * np.log10(smallset_filtered_results[:, 0])
# # set a threshold for spike rate
# smallset_filtered_results = smallset_filtered_results[threshold_spike_rate <= smallset_filtered_results[:, 1]]
# smallset_filtered_results = smallset_filtered_results[max_spike_threshold_rate >= smallset_filtered_results[:, 1]]
# smallset_filtered_results = smallset_filtered_results[smallset_filtered_results[:, 2] <= max_signal_length]
# smallset_filtered_results = smallset_filtered_results[smallset_filtered_results[:, 2] >= min_signal_length]
# df1 = pd.DataFrame(smallset_filtered_results[:, [0, 1]],
#                    columns=['SNR in DB', 'spike rate as a fraction of Nyquist rate = 44.1kHz'])
# # sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz', data=df1, scatter_kws={"s": 3},
# #            line_kws={'color': 'purple'})
#
# reports_csv = '../csvresults/cbpdn_coconut.csv'
# length_col_index = 4
# sparse_code_results = read_numpy_array_from_csv(reports_csv)
# filtered_sparse_code_results = sparse_code_results[:, [1, 2, length_col_index]]
# filtered_sparse_code_results = filtered_sparse_code_results[0 <= filtered_sparse_code_results[:, 0]]
# filtered_sparse_code_results = filtered_sparse_code_results[filtered_sparse_code_results[:, 0] < 1]
# filtered_sparse_code_results[:, 0] = -10 * np.log10(filtered_sparse_code_results[:, 0])
#
# # set a threshold for spike rate
# filtered_sparse_code_results = filtered_sparse_code_results[threshold_spike_rate <= filtered_sparse_code_results[:, 1]]
# filtered_sparse_code_results = filtered_sparse_code_results[
#     max_spike_threshold_rate >= filtered_sparse_code_results[:, 1]]
# filtered_sparse_code_results = filtered_sparse_code_results[max_signal_length >= filtered_sparse_code_results[:, 2]]
# filtered_sparse_code_results = filtered_sparse_code_results[min_signal_length <= filtered_sparse_code_results[:, 2]]
# df2 = pd.DataFrame(filtered_sparse_code_results[:, [0, 1]],
#                    columns=['SNR in DB', 'spike rate as a fraction of Nyquist rate = 44.1kHz'])
#
# concatenated_df = pd.concat([df1.assign(dataset='Our framework'),
#                              df2.assign(dataset='CBPDN')])
# sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz',
#            data=concatenated_df, hue='dataset', scatter_kws={"s": 3})
# print('done')
# #
# ############################################################################################
# # following piece of code is used to plot the results of the comparative study between     #
# # runtimes of our technique and advanced sporco based convolutional sparse code techniques #
# ############################################################################################
#
# reports_csv = '../csvresults/recons_reports_csc_comparison_10_kernel_lime.csv'
# # TODO: set this properly
# time_col_name = 'Length of the audio snippet in ms'
# processing_time_col_name = 'Processing time in second'
# time_col_index = 8
# length_col_index = 10
# small_set_results_with_time = read_numpy_array_from_csv(reports_csv)
# smallset_filtered_results_with_time = small_set_results_with_time[:, [1, 4, time_col_index, length_col_index]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[0 <= smallset_filtered_results_with_time[:, 0]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[smallset_filtered_results_with_time[:, 0] < 1]
# smallset_filtered_results_with_time[:, 0] = -10 * np.log10(smallset_filtered_results_with_time[:, 0])
# smallset_filtered_results_with_time[:, 1] = smallset_filtered_results_with_time[:, 1]
# # set a threshold for spike rate
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[threshold_spike_rate <= smallset_filtered_results_with_time[:, 1]]
# smallset_filtered_results_with_time = \
#     smallset_filtered_results_with_time[max_spike_threshold_rate >= smallset_filtered_results_with_time[:, 1]]
#
# smallset_filtered_results_with_time = smallset_filtered_results_with_time[min_signal_length
#                                                                           <= smallset_filtered_results_with_time[:, 3]]
# smallset_filtered_results_with_time = smallset_filtered_results_with_time[max_signal_length
#                                                                           >= smallset_filtered_results_with_time[:, 3]]
# smallset_filtered_results_with_time = smallset_filtered_results_with_time[:, [2, 3]]
# smallset_filtered_results_with_time[:, 1] = smallset_filtered_results_with_time[:, 1] / 44.1
# df1 = pd.DataFrame(smallset_filtered_results_with_time,
#                    columns=[processing_time_col_name, time_col_name])
# # sns.lmplot(x='SNR in DB', y='spike rate as a fraction of Nyquist rate = 44.1kHz', data=df1, scatter_kws={"s": 3},
# #            line_kws={'color': 'purple'})
#
# reports_csv = '../csvresults/cbpdn_coconut.csv'
# # minimum spike rate to be considered as a fraction of Nq rate
# time_col_index = 3
# length_col_index = 4
# sparse_code_results_with_time = read_numpy_array_from_csv(reports_csv)
# filtered_sparse_code_results_with_time = sparse_code_results_with_time[:, [1, 2, time_col_index, length_col_index]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[0 <= filtered_sparse_code_results_with_time[:, 0]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[filtered_sparse_code_results_with_time[:, 0] < 1]
# filtered_sparse_code_results_with_time[:, 0] = -10 * np.log10(filtered_sparse_code_results_with_time[:, 0])
# # set a threshold for spike rate
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[threshold_spike_rate <= filtered_sparse_code_results_with_time[:, 1]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[max_spike_threshold_rate >= filtered_sparse_code_results_with_time[:, 1]]
#
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[min_signal_length <= filtered_sparse_code_results_with_time[:, 3]]
# filtered_sparse_code_results_with_time = \
#     filtered_sparse_code_results_with_time[max_signal_length >= filtered_sparse_code_results_with_time[:, 3]]
# filtered_sparse_code_results_with_time = filtered_sparse_code_results_with_time[:, [2, 3]]
# filtered_sparse_code_results_with_time[:, 1] = filtered_sparse_code_results_with_time[:, 1] / 44.1
# df2 = pd.DataFrame(filtered_sparse_code_results_with_time,
#                    columns=[processing_time_col_name, time_col_name])
#
# concatenated_df = pd.concat([df1.assign(dataset='Our framework'),
#                              df2.assign(dataset='CBPDN')])
# sns.lmplot(x=time_col_name, y=processing_time_col_name,
#            data=concatenated_df, hue='dataset', scatter_kws={"s": 3})
#
# print('done')
