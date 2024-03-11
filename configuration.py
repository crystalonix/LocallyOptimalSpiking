import logging

audio_filepath = 'audio_text/'
training_sample_folder_path = './../train_curated/'
training_sub_sample_folder_path = './../audio_train_subsample/'

log_file = 'log.txt'
logging_level = logging.DEBUG

NUMBER_OF_BSPLINE_COMPONENTS = 200
SCHUR_POWER = -1.0
upsample_factor = 10

# ahp parameters
ahp_period = 1000.0 * upsample_factor
ahp_high_value = 5.0e-5 * upsample_factor
# denotes whether the ahp drop should consider only one last spike
single_ahp = True

# lateral inhibition parameters
lateral_inhibition_exponent = 2
lateral_neighborhood = 10

spiking_threshold = 5.0e-7
actual_sampling_rate = 44100.0
sampling_rate = actual_sampling_rate * upsample_factor
snippet_length = 10000
# while reconstructing in piecewise mode,
# this decides how much overlap is needed with the previous piece
signal_interleaving_length = 7000
# this is the max spike count beyond which spikes will not be produced in a single run
max_spike_count = 1000000
# this is the max spike rate at which grid search will stop
max_allowed_spikes = 15000

quantized_threshold_transmission = False
precondition_mode = False
z_score_by_residual_norm = False
calculate_recons_coeffs = False
direct_invert_p = True
testing = True
windowing_batch_mode = True
parallel_convolution = True

variable_threshold = True
threshold_reduction_factor = 10

number_of_threads = 40
verbose = False
debug = False
compute_time = True

mode = "expanded"
window_size = 10
window_spike_batch_size = 500
# 'compressed'
