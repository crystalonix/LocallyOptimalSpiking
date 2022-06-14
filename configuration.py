NUMBER_OF_BSPLINE_COMPONENTS = 200
SCHUR_POWER = -1.0
upsample_factor = 10
ahp_period = 20.0 * upsample_factor
ahp_high_value = 1000.0 * upsample_factor
spiking_threshold = 5.0e-6
actual_sampling_rate = 44100.0
sampling_rate = actual_sampling_rate * upsample_factor

precondition_mode = False
z_score_by_residual_norm = True
direct_invert_p = True
testing = True

number_of_threads = 10
verbose = False
debug = False
mode = "expanded"
window_size = 10
# 'compressed'
