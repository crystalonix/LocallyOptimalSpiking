import os
import random
import shutil
from scipy.io import wavfile
import plot_utils
import configuration
import numpy as np

MAX_WAV_16_AMPLITUDE_VAL = 32768
WAV_FILE_DATA_TYPE = np.int16


#################### generate sub sample wav files' folder #################
# random_subsample_size = 10
# all_files = os.listdir(training_sample_folder_path)
# random.shuffle(all_files)
#
# for i, f in enumerate(all_files[:random_subsample_size]):
#     shutil.copyfile(training_sample_folder_path+f, training_sub_sample_folder_path + str(i) + '.wav')


def wav_to_float_data(file_name):
    samp_rate, data = wavfile.read(file_name)
    data = data / MAX_WAV_16_AMPLITUDE_VAL
    return data


def store_float_date_to_wav(file_name, float_data_array, rate=configuration.actual_sampling_rate):
    data_int = np.array(float_data_array * MAX_WAV_16_AMPLITUDE_VAL, dtype=WAV_FILE_DATA_TYPE)
    wavfile.write(file_name, int(rate), data_int)


# def get_full_signal(sample_number):
#     file_name = training_sub_sample_folder_path + str(sample_number) + '.wav'
#     return wav_to_float_data()

# all_files = os.listdir(training_sub_sample_folder_path)
# total_number_of_wav_files = 2
# for i in range(total_number_of_wav_files):
#     file_name = training_sub_sample_folder_path + str(i) + '.wav'
#     data = wav_to_float_data(file_name)
#     # samp_rate, data = wavfile.read(file_name)
#     store_float_date_to_wav(training_sub_sample_folder_path + str(i) + '-recons.wav', data)
#     plot_utils.plot_function(data)
