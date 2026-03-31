from dataclasses import dataclass, field
from features.categories import FeatureCategory

@dataclass
class FeatureExtractionConfig:
    """
    Configuration centralisée de l'extraction de features.
    """

    bands: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "delta": (1.0, 4.0),
        "theta": (5.0, 8.0),
        "alpha": (9.0, 13.0),
        "beta": (14.0, 30.0),
        "gamma": (31.0, 48.0),
        "full": (0.5, 48.0),
    })
    wamp_threshold: float = 0.01
    entropy_m: int = 2
    entropy_r_factor: float = 0.2
    permutation_order: int = 3
    permutation_delay: int = 1

    wavelet: str = "db1"
    wavelet_level: int = 1
    
    spectral_flux_segment_sec: float = 1.0
    psd_time_halfbandwidth_product:float = 2.5
    higuchi_kmax: int = 10
    lyapunov_emb_dim: int = 6
    lyapunov_tau: int = 1
    lyapunov_max_t: int = 20
    corr_dim_emb_dim: int = 3
    corr_dim_tau: int = 1

    ppc_epoch_duration:float = 2.0
    ppc_epoch_overlap:float = 0.0
    ppc_mode:str="multitaper"
    ppc_n_jobs:int=1

    categories_to_extract : list[FeatureCategory] = None