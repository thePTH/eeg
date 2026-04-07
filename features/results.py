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
from eeg.signal import SampledSignal
from features.definitions.base import EEGExtractedFeature


@dataclass(frozen=True)
class FeatureExtractionResult:
    """
    Résultat de l'extraction des features scalaires par canal.

    Cette classe ne contient volontairement **que** les features métier
    utilisées dans le tableau principal des features. La PSD et la PPC ont
    désormais leurs propres classes de résultat.
    """

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    features_dico: dict[SampledSignal, list[EEGExtractedFeature]]

    @property
    def config(self) -> FeatureExtractionConfig:
        return self.extraction_config

    @property
    def feature_names(self) -> list[str]:
        if not self.features_dico:
            return []
        extracted_features = next(iter(self.features_dico.values()))
        return [feature.name for feature in extracted_features]

    @property
    def signal_names(self) -> list[str]:
        return [signal.name for signal in self.features_dico.keys()]

    @property
    def dico(self) -> dict[str, dict[str, float]]:
        values_dico: dict[str, dict[str, float]] = {}
        for signal, extracted_features in self.features_dico.items():
            values_dico[signal.name] = {
                extracted_feature.name: float(extracted_feature.value)
                for extracted_feature in extracted_features
            }
        return values_dico

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.dico, orient="index")

    def values(self, feature_name: str) -> list[float]:
        if feature_name not in self.dataframe.columns:
            raise KeyError(f"Unknown feature '{feature_name}'.")
        return self.dataframe[feature_name].astype(float).tolist()

    def series(self, feature_name: str) -> pd.Series:
        """
        Retourne une série indexée par nom de canal pour une feature donnée.
        """
        if feature_name not in self.dataframe.columns:
            raise KeyError(f"Unknown feature '{feature_name}'.")
        return self.dataframe[feature_name].astype(float)

    def describe_feature(self, feature_name: str) -> pd.Series:
        """
        Statistiques descriptives simples pour une feature.
        """
        return self.series(feature_name).describe()

    def plot_feature_bar(
        self,
        feature_name: str,
        sort: bool = False,
        figsize: tuple[int, int] = (12, 4),
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Barplot des valeurs d'une feature sur les canaux.
        """
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
        """
        Topomap MNE pour visualiser une feature scalaire sur le scalp.
        """
        values = self.series(feature_name)

        ch_names = list(self.eeg.info["ch_names"])
        if list(values.index) != ch_names:
            values = values.reindex(ch_names)

        fig, ax = plt.subplots(figsize=figsize)
        mne.viz.plot_topomap(
            values.to_numpy(dtype=float),
            self.eeg.info,
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
    """
    Résultat PSD exposé uniquement par bande et par canal.

    Structure interne:
    {
        "Fp1": {"delta": ..., "theta": ..., ...},
        "Fp2": {...},
        ...
    }
    """

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    band_powers_by_signal: dict[str, dict[str, float]]

    @property
    def config(self) -> FeatureExtractionConfig:
        return self.extraction_config

    @property
    def signal_names(self) -> list[str]:
        return list(self.band_powers_by_signal.keys())

    @property
    def band_names(self) -> list[str]:
        if not self.band_powers_by_signal:
            return []
        first_signal = next(iter(self.band_powers_by_signal.values()))
        return list(first_signal.keys())

    @property
    def dico(self) -> dict[str, dict[str, float]]:
        return {
            signal_name: {band_name: float(value) for band_name, value in band_dict.items()}
            for signal_name, band_dict in self.band_powers_by_signal.items()
        }

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.dico, orient="index")

    def band_power(self, signal_name: str, band_name: str) -> float:
        return float(self.band_powers_by_signal[signal_name][band_name])

    def band_series(self, band_name: str) -> pd.Series:
        """
        Retourne la puissance PSD d'une bande sur tous les canaux.
        """
        if band_name not in self.band_names:
            raise KeyError(f"Unknown PSD band '{band_name}'. Available bands: {self.band_names}")
        return self.dataframe[band_name].astype(float)

    def describe_band(self, band_name: str) -> pd.Series:
        """
        Statistiques descriptives simples pour une bande PSD.
        """
        return self.band_series(band_name).describe()

    def to_serializable_dict(self) -> dict[str, dict[str, float]]:
        return self.dico

    def plot_band_bar(
        self,
        band_name: str,
        sort: bool = False,
        figsize: tuple[int, int] = (12, 4),
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Barplot des puissances PSD pour une bande.
        """
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
        """
        Topomap de la puissance PSD d'une bande.
        """
        values = self.band_series(band_name)

        ch_names = list(self.eeg.info["ch_names"])
        if list(values.index) != ch_names:
            values = values.reindex(ch_names)

        fig, ax = plt.subplots(figsize=figsize)
        mne.viz.plot_topomap(
            values.to_numpy(dtype=float),
            self.eeg.info,
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
    """
    Résultat PPC exposé uniquement par bande.

    Structure interne:
    {
        "delta": np.ndarray (n_channels, n_channels),
        "theta": np.ndarray (n_channels, n_channels),
        ...
    }
    """

    eeg: EEGProcessedData
    extraction_config: FeatureExtractionConfig
    matrices_by_band: dict[str, np.ndarray]

    @property
    def config(self) -> FeatureExtractionConfig:
        return self.extraction_config

    @property
    def band_names(self) -> list[str]:
        return list(self.matrices_by_band.keys())

    @property
    def channel_names(self) -> list[str]:
        return list(self.eeg.info["ch_names"])

    def matrix(self, band_name: str) -> np.ndarray:
        if band_name not in self.matrices_by_band:
            raise KeyError(
                f"Unknown PPC band '{band_name}'. Available bands: {self.band_names}"
            )
        return np.asarray(self.matrices_by_band[band_name], dtype=float)

    def to_serializable_dict(self) -> dict[str, list[list[float]]]:
        return {
            band_name: self.matrix(band_name).tolist()
            for band_name in self.band_names
        }

    def describe_band(self, band_name: str) -> pd.Series:
        """
        Statistiques descriptives sur les arêtes de la matrice PPC.
        Par défaut, on prend uniquement le triangle supérieur sans diagonale.
        """
        mat = self.matrix(band_name)
        ii, jj = np.triu_indices_from(mat, k=1)
        values = mat[ii, jj]
        return pd.Series(values).describe()

    def mean_connectivity(self, band_name: str, include_diagonal: bool = False) -> float:
        """
        Moyenne de connectivité d'une bande.
        """
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
        rows: list[dict[str, Any]] = []
        channel_names = self.channel_names

        for band_name in self.band_names:
            mat = self.matrix(band_name)
            n_channels = mat.shape[0]

            if include_symmetric:
                iterator = (
                    (i, j)
                    for i in range(n_channels)
                    for j in range(n_channels)
                    if include_diagonal or i != j
                )
            else:
                k = 0 if include_diagonal else 1
                ii, jj = np.triu_indices(n_channels, k=k)
                iterator = zip(ii.tolist(), jj.tolist())

            for i, j in iterator:
                rows.append(
                    {
                        "band": band_name,
                        "seed": channel_names[i],
                        "target": channel_names[j],
                        "value": float(mat[i, j]),
                    }
                )

        return pd.DataFrame(rows)

    def plot_band_matrix(
        self,
        band_name: str,
        figsize: tuple[int, int] = (8, 6),
        cmap: str = "viridis",
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Heatmap simple de la matrice PPC pour une bande.
        """
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

    def plot_band_graph(
        self,
        band_name: str,
        threshold: float | None = None,
        percentile: float | None = 90.0,
        figsize: tuple[int, int] = (8, 8),
        node_size: int = 500,
    ):
        """
        Visualisation graphe de la connectivité PPC.

        Paramètres
        ----------
        band_name:
            Bande à afficher.
        threshold:
            Seuil absolu sur la valeur PPC. Si fourni, prioritaire.
        percentile:
            Si `threshold` est None, on conserve seulement les arêtes
            au-dessus de ce percentile.
        """
        mat = self.matrix(band_name).copy()
        ch_names = self.channel_names
        n = mat.shape[0]

        ii, jj = np.triu_indices_from(mat, k=1)
        edge_values = mat[ii, jj]

        if threshold is None:
            if percentile is None:
                threshold = float(np.min(edge_values))
            else:
                threshold = float(np.percentile(edge_values, percentile))

        graph = nx.Graph()
        for ch in ch_names:
            graph.add_node(ch)

        for i, j in zip(ii, jj):
            value = float(mat[i, j])
            if value >= threshold:
                graph.add_edge(ch_names[i], ch_names[j], weight=value)

        pos = nx.circular_layout(graph)

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, ax=ax)
        nx.draw_networkx_labels(graph, pos, font_size=9, ax=ax)

        edges = list(graph.edges(data=True))
        if edges:
            widths = [2 + 4 * edge_data["weight"] for _, _, edge_data in edges]
            nx.draw_networkx_edges(graph, pos, width=widths, alpha=0.7, ax=ax)

        ax.set_title(
            f"PPC graph - {band_name}\n"
            f"(threshold={threshold:.4f}, edges={graph.number_of_edges()})"
        )
        ax.axis("off")
        plt.tight_layout()
        return fig, ax