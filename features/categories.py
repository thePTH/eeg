from enum import Enum


class FeatureCategory(Enum):
    COMPLEXITY = "complexity"
    ENTROPY = "entropy"
    POWER_RATIO = "power_ratio"
    SPECTRAL = "spectral"
    TEMPORAL = "temporal"
    WAVELET = "wavelet"