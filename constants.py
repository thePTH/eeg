EPS = 0

N_SUBJECTS = 88
DATA_ROOT_PATH_NAME = "data"

BANDPASS = (0.5, 48)
CROP_TMIN = 10
CROP_TMAX = 70

ASR_CUTOFF = 4
ASR_BLOCKSIZE = 100
ASR_WIN_LEN = 1
ASR_WIN_OVERLAP = 0.66
ASR_MAX_DROPOUT_FRACTION = 0.1
ASR_MIN_CLEAN_FRACTION = 0.25
ASR_MAX_BAD_CHANS = 0.1


GLOBAL_DETREND_ORDER = 1
LOCAL_DETREND_WINDOW_SEC = 0.06
LOCAL_DETREND_STEP_SEC = 0.015



HAMPEL_Q = 1000
HAMPEL_WINDOW_SIZE = 2 * HAMPEL_Q + 1
HAMPEL_N_SIGMA = 3.0

DEFAULT_FEATURE_ORDER = [
    # temporelles
    "variance",
    "skewness",
    "kurtosis",
    "shape_factor",
    "peak_amplitude",
    "impulse_factor",
    "crest_factor",
    "clearance_factor",
    "willison_amplitude",
    "zero_crossing_rate",
    # entropies
    "sample_entropy",
    "approximate_entropy",
    "permutation_entropy",
    "state_space_correlation_entropy",
    # complexité
    "correlation_dimension",
    "higuchi_fractal_dimension",
    "katz_fractal_dimension",
    "lyapunov_exponent",
    "hurst_exponent",
    "lempel_ziv_complexity",
    "hjorth_activity",
    "hjorth_mobility",
    "hjorth_complexity",
    # spectral
    "alpha_dominant_frequency",
    "gamma_dominant_frequency",
    "spectral_rolloff",
    "spectral_centroid",
    "spectral_spread",
    "spectral_flux",
    "spectral_skewness",
    "spectral_kurtosis",
    # ratios de puissance
    "theta_beta_ratio",
    "theta_alpha_ratio",
    "gamma_alpha_ratio",
    "spectral_power_ratio",
    # wavelets
    "wavelet_energy_approximate",
    "wavelet_energy_detail",
    "relative_wavelet_energy",
    "wavelet_packet_energy_approximate",
    "wavelet_packet_energy_detail",
    "relative_wavelet_packet_energy",
]

