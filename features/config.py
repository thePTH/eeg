from __future__ import annotations

from dataclasses import dataclass, field

from features.categories import FeatureCategory


@dataclass
class FeatureExtractionConfig:
    """
    Central configuration for EEG feature extraction.

    Notes
    -----
    Mutable defaults are created with ``default_factory`` to avoid
    shared state across configuration instances.
    """

    #: Frequency bands used throughout the spectral analysis.
    bands: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "delta": (1.0, 4.0),
            "theta": (5.0, 8.0),
            "alpha": (9.0, 13.0),
            "beta": (14.0, 30.0),
            "gamma": (31.0, 48.0),
            "full": (0.5, 48.0),
        }
    )

    #: Threshold used by the Willison Amplitude feature.
    wamp_threshold: float = 0.01

    #: Embedding dimension for approximate/sample entropy.
    entropy_m: int = 2

    #: Tolerance factor (r = factor × signal standard deviation).
    entropy_r_factor: float = 0.2

    #: Embedding order for permutation entropy.
    permutation_order: int = 3

    #: Time delay for permutation entropy.
    permutation_delay: int = 1

    #: Mother wavelet used for wavelet decomposition.
    wavelet: str = "db1"

    #: Maximum decomposition level for wavelet analysis.
    wavelet_level: int = 1

    #: Segment duration (seconds) used to compute spectral flux.
    spectral_flux_segment_sec: float = 1.0

    #: Time-half bandwidth product used for multitaper PSD estimation.
    psd_time_halfbandwidth_product: float = 2.5

    #: Maximum scale parameter for Higuchi fractal dimension.
    higuchi_kmax: int = 10

    #: Embedding dimension for Lyapunov exponent estimation.
    lyapunov_emb_dim: int = 6

    #: Delay used in Lyapunov embedding.
    lyapunov_tau: int = 1

    #: Maximum prediction horizon for Lyapunov estimation.
    lyapunov_max_t: int = 20

    #: Embedding dimension for correlation dimension estimation.
    corr_dim_emb_dim: int = 3

    #: Delay used in correlation dimension embedding.
    corr_dim_tau: int = 1

    #: Epoch duration (seconds) for PPC computation.
    ppc_epoch_duration: float = 2.0

    #: Overlap ratio between consecutive PPC epochs.
    ppc_epoch_overlap: float = 0.0

    #: Spectral estimation method used for PPC.
    ppc_mode: str = "multitaper"

    #: Number of parallel workers used during PPC computation.
    ppc_n_jobs: int = 1

    #: Optional subset of feature categories to extract.
    #: If ``None``, all registered categories are extracted.
    categories_to_extract: list[FeatureCategory] | None = None