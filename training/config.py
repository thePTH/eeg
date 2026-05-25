from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str = "eeg_neurosymbolic"

    random_seed: int = 42

    dataset_folder: str = "computed_features/dethamp"
    dataset_name: str = "raw_data"

    target_health_states: tuple[str, ...] = ("AD", "CN")

    feature_family_names: tuple[str, ...] = (
        "theta_alpha_ratio",
        "theta_beta_ratio",
        "gamma_alpha_ratio",
        "spectral_power_ratio",
        "spectral_centroid",
        "spectral_spread",
        "spectral_flux",
        "spectral_rolloff",
        "alpha_dominant_frequency",
        "gamma_dominant_frequency",
        "relative_wavelet_energy",
        "relative_wavelet_packet_energy",
        "alpha",
        "beta",
        "gamma",
    )

    test_size: float = 0.2
    val_size: float = 0.3

    n_rules_to_keep: int = 2

    batch_size: int = 8
    preprocessing_mode: str = "mtdnet"
    use_mtdnet_split: bool = True

    epochs: int = 50
    lambda_logic: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4

    macro_aggregation_method: str = "mean_probability"
    supervised_loss_compute_method: str = "micro_bce"