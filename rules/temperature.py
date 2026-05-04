import numpy as np
from features.dataset import SelectedFeaturesDataset

import torch

class TemperatureFeatureMapping:
    def __init__(self, dico: dict[str, float]):
        self.dico = dico
        self.feature_to_index = {
            feature_name: index
            for index, feature_name in enumerate(dico.keys())
        }

    def __call__(self, feature_name: str) -> float:
        return self.dico[feature_name]

    def index(self, feature_name: str) -> int:
        return self.feature_to_index[feature_name]

    def vector(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Retourne un tensor [n_features] compatible PyTorch.
        """
        return torch.tensor(
            list(self.dico.values()),
            dtype=dtype,
            device=device,
        )
    


class TemperatureFeatureMappingFactory:

    @staticmethod
    def build(
        dataset: SelectedFeaturesDataset,
        c: float = 0.1,
        min_tau: float = 1e-3,
    ) -> TemperatureFeatureMapping:
        """
        Construit un mapping feature_name -> température tau.

        La température est calculée avec :
            tau = c * IQR(feature)

        où :
            IQR = percentile_75 - percentile_25

        Un minimum est appliqué pour éviter tau = 0.
        """

        X = dataset.X

        q75 = X.quantile(0.75)
        q25 = X.quantile(0.25)

        iqr = q75 - q25

        tau_values = c * iqr
        tau_values = tau_values.clip(lower=min_tau)

        temperature_by_feature = {
            feature_name: float(tau_values[feature_name])
            for feature_name in X.columns
        }

        return TemperatureFeatureMapping(temperature_by_feature)