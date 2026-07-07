from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import mne
import networkx as nx
import numpy as np
import pandas as pd

from eeg.data import EEGProcessedData
from features.config import FeatureExtractionConfig
from features.definitions.base import EEGExtractedFeature


@dataclass(frozen=True)
class FeatureExtractionResult:
    """
    Result of scalar channel-wise feature extraction.

    This class does not store raw signals. It only keeps:
    - extracted feature values;
    - a reference to the processed EEG object;
    - a serializable snapshot of ``mne.Info``.

    The ``eeg_info_dico`` snapshot allows the result to remain usable even if
    the EEG object has later been unloaded.
    """

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    features_dico: dict[str, list[EEGExtractedFeature]]
    eeg_info_dico: dict[str, Any]

    @property
    def config(self) -> FeatureExtractionConfig:
        """Return the feature extraction configuration."""
        return self.extraction_config

    @property
    def eeg_info(self) -> mne.Info:
        """Return the reconstructed MNE Info object."""
        return mne.Info.from_json_dict(self.eeg_info_dico)

    @property
    def feature_names(self) -> list[str]:
        """Return the extracted feature names."""
        if not self.features_dico:
            return []

        extracted_features = next(iter(self.features_dico.values()))

        return [feature.name for feature in extracted_features]

    @property
    def signal_names(self) -> list[str]:
        """Return the signal names."""
        return list(self.features_dico.keys())

    @property
    def dico(self) -> dict[str, dict[str, float]]:
        """Return extracted features as a nested dictionary."""
        values_dico: dict[str, dict[str, float]] = {}

        for signal_name, extracted_features in self.features_dico.items():
            values_dico[signal_name] = {
                extracted_feature.name: float(extracted_feature.value)
                for extracted_feature in extracted_features
            }

        return values_dico

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return extracted features as a DataFrame."""
        return pd.DataFrame.from_dict(self.dico, orient="index")

    def values(self, feature_name: str) -> list[float]:
        """Return all channel values for a feature as a list."""
        if feature_name not in self.dataframe.columns:
            raise KeyError(f"Unknown feature '{feature_name}'.")

        return self.dataframe[feature_name].astype(float).tolist()

    def series(self, feature_name: str) -> pd.Series:
        """Return all channel values for a feature as a Series."""
        if feature_name not in self.dataframe.columns:
            raise KeyError(f"Unknown feature '{feature_name}'.")

        return self.dataframe[feature_name].astype(float)

    def describe_feature(self, feature_name: str) -> pd.Series:
        """Return descriptive statistics for one feature."""
        return self.series(feature_name).describe()

    def to_serializable_dict(self) -> dict[str, dict[str, float]]:
        """Return a JSON-serializable representation of extracted features."""
        return self.dico

    def plot_feature_bar(
        self,
        feature_name: str,
        sort: bool = False,
        figsize: tuple[int, int] = (12, 4),
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot channel-wise values of one feature as a bar chart."""
        values = self.series(feature_name)

        if sort:
            values = values.sort_values(ascending=False)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        values.plot(kind="bar", ax=ax)
        ax.set_title(f"Feature '{feature_name}' by channel")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        return ax

    def plot_feature_topomap(
        self,
        feature_name: str,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (6, 5),
        show_names: bool = False,
    ):
        """Plot channel-wise values of one feature as an EEG topomap."""
        values = self.series(feature_name)

        ch_names = list(self.eeg_info["ch_names"])

        if list(values.index) != ch_names:
            values = values.reindex(ch_names)

        fig, ax = plt.subplots(figsize=figsize)
        mne.viz.plot_topomap(
            values.to_numpy(dtype=float),
            self.eeg_info,
            axes=ax,
            show=False,
            cmap=cmap,
            names=ch_names if show_names else None,
        )
        ax.set_title(f"Topomap - {feature_name}")
        plt.tight_layout()

        return fig, ax


@dataclass(frozen=True)
class PSDBandExtractionResult:
    """PSD extraction result exposed by frequency band and channel."""

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    band_powers_by_signal: dict[str, dict[str, float]]
    eeg_info_dico: dict[str, Any]

    @property
    def config(self) -> FeatureExtractionConfig:
        """Return the feature extraction configuration."""
        return self.extraction_config

    @property
    def eeg_info(self) -> mne.Info:
        """Return the reconstructed MNE Info object."""
        return mne.Info.from_json_dict(self.eeg_info_dico)

    @property
    def signal_names(self) -> list[str]:
        """Return the signal names."""
        return list(self.band_powers_by_signal.keys())

    @property
    def band_names(self) -> list[str]:
        """Return the PSD band names."""
        if not self.band_powers_by_signal:
            return []

        first_signal = next(iter(self.band_powers_by_signal.values()))

        return list(first_signal.keys())

    @property
    def dico(self) -> dict[str, dict[str, float]]:
        """Return PSD band powers as a nested dictionary."""
        return {
            signal_name: {
                band_name: float(value)
                for band_name, value in band_dict.items()
            }
            for signal_name, band_dict in self.band_powers_by_signal.items()
        }

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return PSD band powers as a DataFrame."""
        return pd.DataFrame.from_dict(self.dico, orient="index")

    def band_power(self, signal_name: str, band_name: str) -> float:
        """Return one PSD band power for one signal."""
        return float(self.band_powers_by_signal[signal_name][band_name])

    def band_series(self, band_name: str) -> pd.Series:
        """Return all channel values for one PSD band as a Series."""
        if band_name not in self.band_names:
            raise KeyError(
                f"Unknown PSD band '{band_name}'. Available bands: {self.band_names}"
            )

        return self.dataframe[band_name].astype(float)

    def describe_band(self, band_name: str) -> pd.Series:
        """Return descriptive statistics for one PSD band."""
        return self.band_series(band_name).describe()

    def to_serializable_dict(self) -> dict[str, dict[str, float]]:
        """Return a JSON-serializable representation of PSD band powers."""
        return self.dico

    def plot_band_bar(
        self,
        band_name: str,
        sort: bool = False,
        figsize: tuple[int, int] = (12, 4),
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot channel-wise PSD band powers as a bar chart."""
        values = self.band_series(band_name)

        if sort:
            values = values.sort_values(ascending=False)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        values.plot(kind="bar", ax=ax)
        ax.set_title(f"PSD band '{band_name}' by channel")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Band power")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        return ax

    def plot_band_topomap(
        self,
        band_name: str,
        cmap: str = "viridis",
        figsize: tuple[int, int] = (6, 5),
        show_names: bool = False,
    ):
        """Plot channel-wise PSD band powers as an EEG topomap."""
        values = self.band_series(band_name)

        ch_names = list(self.eeg_info["ch_names"])

        if list(values.index) != ch_names:
            values = values.reindex(ch_names)

        fig, ax = plt.subplots(figsize=figsize)
        mne.viz.plot_topomap(
            values.to_numpy(dtype=float),
            self.eeg_info,
            axes=ax,
            show=False,
            cmap=cmap,
            names=ch_names if show_names else None,
        )
        ax.set_title(f"PSD topomap - {band_name}")
        plt.tight_layout()

        return fig, ax


@dataclass(frozen=True)
class PPCBandExtractionResult:
    """PPC extraction result exposed by frequency band."""

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    matrices_by_band: dict[str, np.ndarray]
    eeg_info_dico: dict[str, Any]

    @property
    def config(self) -> FeatureExtractionConfig:
        """Return the feature extraction configuration."""
        return self.extraction_config

    @property
    def eeg_info(self) -> mne.Info:
        """Return the reconstructed MNE Info object."""
        return mne.Info.from_json_dict(self.eeg_info_dico)

    @property
    def band_names(self) -> list[str]:
        """Return the PPC band names."""
        return list(self.matrices_by_band.keys())

    @property
    def channel_names(self) -> list[str]:
        """Return the EEG channel names."""
        return list(self.eeg_info["ch_names"])

    def matrix(self, band_name: str) -> np.ndarray:
        """Return the PPC matrix associated with one frequency band."""
        if band_name not in self.matrices_by_band:
            raise KeyError(
                f"Unknown PPC band '{band_name}'. Available bands: {self.band_names}"
            )

        return np.asarray(self.matrices_by_band[band_name], dtype=float)

    def to_serializable_dict(self) -> dict[str, list[list[float]]]:
        """Return a JSON-serializable representation of PPC matrices."""
        return {
            band_name: self.matrix(band_name).tolist()
            for band_name in self.band_names
        }

    def describe_band(self, band_name: str) -> pd.Series:
        """Return descriptive statistics for upper-triangular PPC values."""
        mat = self.matrix(band_name)
        ii, jj = np.triu_indices_from(mat, k=1)
        values = mat[ii, jj]

        return pd.Series(values).describe()

    def mean_connectivity(self, band_name: str, include_diagonal: bool = False) -> float:
        """Return the mean PPC connectivity for one frequency band."""
        mat = self.matrix(band_name)

        if include_diagonal:
            return float(np.mean(mat))

        ii, jj = np.triu_indices_from(mat, k=1)

        return float(np.mean(mat[ii, jj]))

    def to_edge_dataframe(
        self,
        include_diagonal: bool = False,
        include_symmetric: bool = False,
    ) -> pd.DataFrame:
        """Convert PPC matrices to an edge-level DataFrame."""
        rows: list[dict[str, Any]] = []

        for band_name in self.band_names:
            mat = self.matrix(band_name)
            n = mat.shape[0]

            for i in range(n):
                for j in range(n):
                    if not include_diagonal and i == j:
                        continue

                    if not include_symmetric and j <= i:
                        continue

                    rows.append(
                        {
                            "band": band_name,
                            "seed": self.channel_names[i],
                            "target": self.channel_names[j],
                            "value": float(mat[i, j]),
                        }
                    )

        return pd.DataFrame(rows)

    def graph(
        self,
        band_name: str,
        threshold: float | None = None,
        use_absolute_value: bool = False,
    ) -> nx.Graph:
        """Build a NetworkX graph from a PPC matrix."""
        mat = self.matrix(band_name)
        G = nx.Graph()

        for ch in self.channel_names:
            G.add_node(ch)

        for i in range(len(self.channel_names)):
            for j in range(i + 1, len(self.channel_names)):
                weight = float(mat[i, j])
                weight_for_threshold = abs(weight) if use_absolute_value else weight

                if threshold is not None and weight_for_threshold < threshold:
                    continue

                G.add_edge(
                    self.channel_names[i],
                    self.channel_names[j],
                    weight=weight,
                )

        return G

    def plot_matrix(
        self,
        band_name: str,
        figsize: tuple[int, int] = (7, 6),
        cmap: str = "viridis",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """Plot the PPC matrix for one frequency band."""
        mat = self.matrix(band_name)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(mat, cmap=cmap)
        ax.set_title(f"PPC matrix - {band_name}")
        ax.set_xticks(range(len(self.channel_names)))
        ax.set_yticks(range(len(self.channel_names)))
        ax.set_xticklabels(self.channel_names, rotation=90)
        ax.set_yticklabels(self.channel_names)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        return ax