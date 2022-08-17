NUMBER_OF_BSPLINE_COMPONENTS = 200
SCHUR_POWER = -1.0
upsample_factor = 10
ahp_period = 20.0 * upsample_factor
ahp_high_value = 1000.0 * upsample_factor
spiking_threshold = 5.0e-6
actual_sampling_rate = 44100.0
sampling_rate = actual_sampling_rate * upsample_factor
# this is the max spike count beyond which spikes will not be produced in a single run
max_spike_count = 1000000
# this is the max spike rate at which grid search will stop
max_allowed_spikes = 15000

precondition_mode = False
z_score_by_residual_norm = False
calculate_recons_coeffs = False
direct_invert_p = True
testing = True

number_of_threads = 10
verbose = False
debug = False
compute_time = True

mode = "expanded"
window_size = 10
# 'compressed'
