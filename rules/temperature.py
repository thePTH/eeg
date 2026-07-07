from __future__ import annotations

import torch

from features.dataset import SelectedFeaturesDataset


class TemperatureFeatureMapping:
    """Mapping between feature names, feature indices, and differentiable temperatures."""

    def __init__(self, dico: dict[str, float]):
        """
        Initialize the temperature mapping.

        Parameters
        ----------
        dico
            Dictionary mapping feature names to their temperature values.
        """
        self.dico = dico
        self.feature_to_index = {
            feature_name: index
            for index, feature_name in enumerate(dico.keys())
        }

    def __call__(self, feature_name: str) -> float:
        """Return the temperature associated with a feature."""
        return self.dico[feature_name]

    def index(self, feature_name: str) -> int:
        """Return the column index associated with a feature."""
        return self.feature_to_index[feature_name]

    def vector(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Return the temperatures as a PyTorch tensor.

        Returns
        -------
        torch.Tensor
            One-dimensional tensor of shape ``[n_features]``.
        """
        return torch.tensor(
            list(self.dico.values()),
            dtype=dtype,
            device=device,
        )


class TemperatureFeatureMappingFactory:
    """Factory used to compute feature-wise temperatures for differentiable rules."""

    @staticmethod
    def build(
        dataset: SelectedFeaturesDataset,
        c: float = 0.1,
        min_tau: float = 1e-3,
    ) -> TemperatureFeatureMapping:
        """
        Build a mapping from feature names to temperatures.

        The temperature associated with each feature is computed as:

            tau = c × IQR(feature)

        where IQR is the interquartile range (75th percentile minus 25th percentile).

        A lower bound is applied to avoid zero temperatures.

        Parameters
        ----------
        dataset
            Dataset containing the selected features.
        c
            Scaling coefficient applied to the interquartile range.
        min_tau
            Minimum allowed temperature.

        Returns
        -------
        TemperatureFeatureMapping
            Mapping between feature names and differentiable temperatures.
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