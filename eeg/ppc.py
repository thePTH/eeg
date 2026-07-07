from dataclasses import dataclass, field
from typing import Optional

from eeg.signal import SpectralBand


@dataclass(frozen=True)
class SignalPPCAnalysisParameters:
    """
    Parameters used to compute Pairwise Phase Consistency (PPC).

    Notes
    -----
    - epoch_duration: duration, in seconds, of each epoch created from the raw signal.
    - epoch_overlap: overlap, in seconds, between consecutive epochs.
    - faverage: if True, averages connectivity values inside each frequency band.
    - mode: spectral mode used by MNE-Connectivity ('multitaper', 'fourier', 'cwt_morlet').
    - mt_adaptive: parameter used when mode='multitaper'.
    - mt_low_bias: parameter used when mode='multitaper'.
    - n_jobs: number of parallel jobs.
    """

    bands: tuple[SpectralBand, ...] = field(
        default_factory=lambda: (
            SpectralBand("delta", 1.0, 4.0),
            SpectralBand("theta", 5.0, 8.0),
            SpectralBand("alpha", 9.0, 13.0),
            SpectralBand("beta", 14.0, 30.0),
            SpectralBand("gamma", 31.0, 48.0),
            SpectralBand("full", 1.0, 48.0),
        )
    )
    epoch_duration: float = 2.0
    epoch_overlap: float = 0.0
    faverage: bool = True
    mode: str = "multitaper"
    mt_adaptive: bool = False
    mt_low_bias: bool = True
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    block_size: int = 1000
    n_jobs: int = 1

    @property
    def band_names(self) -> list[str]:
        """Return the names of the configured frequency bands."""
        return [band.name for band in self.bands]

    @property
    def fmin(self) -> tuple[float, ...]:
        """Return the lower frequency bounds of all bands."""
        return tuple(band.fmin for band in self.bands)

    @property
    def fmax(self) -> tuple[float, ...]:
        """Return the upper frequency bounds of all bands."""
        return tuple(band.fmax for band in self.bands)


from dataclasses import dataclass
from functools import cached_property

import numpy as np
from mne_connectivity import Connectivity

from eeg.data import EEGProcessedData


@dataclass(frozen=True)
class SignalPPCAnalysisResult:
    """
    Result of a PPC computation.

    Exposes:
    - eeg: EEGProcessedData object used for the computation.
    - params: PPC computation parameters.
    - connectivity: MNE-Connectivity object.
    - band matrices: accessible through band_matrix(...), theta_matrix, etc.

    Convention used here:
    - each PPC matrix is symmetric;
    - the diagonal is set to 0.0.
    """

    eeg: EEGProcessedData
    params: "SignalPPCAnalysisParameters"
    connectivity: Connectivity

    @property
    def channel_names(self) -> list[str]:
        """Return the channel names stored in the connectivity object."""
        return list(self.connectivity.names)

    @property
    def n_channels(self) -> int:
        """Return the number of EEG channels."""
        return len(self.channel_names)

    @property
    def band_names(self) -> list[str]:
        """Return the frequency band names."""
        return self.params.band_names

    @cached_property
    def channel_name_to_index(self) -> dict[str, int]:
        """Return a mapping from channel names to channel indices."""
        return {name: idx for idx, name in enumerate(self.channel_names)}

    @cached_property
    def dense_data_raw(self) -> np.ndarray:
        """
        Return the dense connectivity data as provided by MNE.

        Expected shape:
        - (n_channels, n_channels, n_bands) if faverage=True;
        - sometimes (n_channels, n_channels) if only one band is used.
        """
        data = self.connectivity.get_data(output="dense")
        data = np.asarray(data, dtype=float)

        if data.ndim == 2:
            data = data[:, :, np.newaxis]

        if data.ndim != 3:
            raise ValueError(
                f"Shape inattendue pour la connectivité dense : {data.shape}. "
                "On attend une shape (n_channels, n_channels, n_bands)."
            )

        if data.shape[0] != self.n_channels or data.shape[1] != self.n_channels:
            raise ValueError(
                "La shape dense n'est pas cohérente avec le nombre de canaux : "
                f"data.shape={data.shape}, n_channels={self.n_channels}"
            )

        return data

    @staticmethod
    def _symmetrize_matrix(mat: np.ndarray, zero_diagonal: bool = True) -> np.ndarray:
        """
        Explicitly symmetrize a matrix.

        Since PPC is a symmetric measure, if information is stored in only one
        half of the matrix, this reconstructs the full matrix.
        """
        work = np.asarray(mat, dtype=float).copy()

        work = np.maximum(work, work.T)

        work = 0.5 * (work + work.T)

        if zero_diagonal:
            np.fill_diagonal(work, 0.0)

        return work

    @cached_property
    def dense_data(self) -> np.ndarray:
        """
        Return the symmetrized dense connectivity data.

        Shape:
            (n_channels, n_channels, n_bands)
        """
        data = self.dense_data_raw.copy()

        for k in range(data.shape[2]):
            data[:, :, k] = self._symmetrize_matrix(data[:, :, k], zero_diagonal=True)

        return data

    @cached_property
    def matrices_by_band(self) -> dict[str, np.ndarray]:
        """Return one PPC matrix per frequency band."""
        n_bands_data = self.dense_data.shape[2]
        n_bands_params = len(self.band_names)

        if n_bands_data != n_bands_params:
            raise ValueError(
                f"Incohérence entre le nombre de bandes dans les données ({n_bands_data}) "
                f"et les paramètres ({n_bands_params}: {self.band_names})."
            )

        return {
            band_name: self.dense_data[:, :, k]
            for k, band_name in enumerate(self.band_names)
        }

    def band_matrix(self, band_name: str) -> np.ndarray:
        """Return the PPC matrix associated with a given frequency band."""
        if band_name not in self.matrices_by_band:
            raise ValueError(
                f"Bande inconnue '{band_name}'. Bandes disponibles : {self.band_names}"
            )

        return self.matrices_by_band[band_name]

    def ppc_value(self, ch_name1: str, ch_name2: str, band_name: str) -> float:
        """
        Return the PPC value between two channels for a given frequency band.

        Parameters
        ----------
        ch_name1 : str
            Name of the first channel.
        ch_name2 : str
            Name of the second channel.
        band_name : str
            Name of the frequency band.

        Returns
        -------
        float
            PPC value.
        """
        if ch_name1 not in self.channel_name_to_index:
            raise ValueError(
                f"Canal inconnu '{ch_name1}'. Canaux disponibles : {self.channel_names}"
            )

        if ch_name2 not in self.channel_name_to_index:
            raise ValueError(
                f"Canal inconnu '{ch_name2}'. Canaux disponibles : {self.channel_names}"
            )

        mat = self.band_matrix(band_name)
        i = self.channel_name_to_index[ch_name1]
        j = self.channel_name_to_index[ch_name2]

        return float(mat[i, j])

    def band_vector_upper(
        self,
        band_name: str,
        include_diagonal: bool = False,
    ) -> np.ndarray:
        """Return the upper-triangular connections of a band matrix as a vector."""
        mat = self.band_matrix(band_name)
        k = 0 if include_diagonal else 1
        i, j = np.triu_indices_from(mat, k=k)

        return mat[i, j]

    def mean_band_connectivity(
        self,
        band_name: str,
        include_diagonal: bool = False,
    ) -> float:
        """Return the mean upper-triangular connectivity for a given band."""
        values = self.band_vector_upper(band_name, include_diagonal=include_diagonal)

        return float(np.mean(values))

    def strongest_connections(
        self,
        band_name: str,
        n: int = 20,
        include_diagonal: bool = False,
    ) -> list[tuple[str, str, float]]:
        """Return the strongest connections for a given frequency band."""
        mat = self.band_matrix(band_name)
        k = 0 if include_diagonal else 1
        ii, jj = np.triu_indices_from(mat, k=k)
        values = mat[ii, jj]

        order = np.argsort(values)[::-1][:n]

        return [
            (self.channel_names[ii[idx]], self.channel_names[jj[idx]], float(values[idx]))
            for idx in order
        ]

    def is_band_symmetric(self, band_name: str, atol: float = 1e-12) -> bool:
        """Return whether a band matrix is symmetric."""
        mat = self.band_matrix(band_name)

        return bool(np.allclose(mat, mat.T, atol=atol))

    def band_summary(self, band_name: str) -> dict[str, float | bool | tuple[int, int]]:
        """Return a compact summary useful for debugging and inspection."""
        mat = self.band_matrix(band_name)

        return {
            "shape": mat.shape,
            "is_symmetric": self.is_band_symmetric(band_name),
            "diag_is_zero": bool(np.allclose(np.diag(mat), 0.0)),
            "min": float(np.min(mat)),
            "max": float(np.max(mat)),
            "mean_upper": self.mean_band_connectivity(band_name),
        }

    @property
    def delta_matrix(self) -> np.ndarray:
        """Return the delta-band PPC matrix."""
        return self.band_matrix("delta")

    @property
    def theta_matrix(self) -> np.ndarray:
        """Return the theta-band PPC matrix."""
        return self.band_matrix("theta")

    @property
    def alpha_matrix(self) -> np.ndarray:
        """Return the alpha-band PPC matrix."""
        return self.band_matrix("alpha")

    @property
    def beta_matrix(self) -> np.ndarray:
        """Return the beta-band PPC matrix."""
        return self.band_matrix("beta")

    @property
    def gamma_matrix(self) -> np.ndarray:
        """Return the gamma-band PPC matrix."""
        return self.band_matrix("gamma")

    @property
    def full_matrix(self) -> np.ndarray:
        """Return the full-band PPC matrix."""
        return self.band_matrix("full")

    @property
    def delta_mean(self) -> float:
        """Return the mean delta-band PPC connectivity."""
        return self.mean_band_connectivity("delta")

    @property
    def theta_mean(self) -> float:
        """Return the mean theta-band PPC connectivity."""
        return self.mean_band_connectivity("theta")

    @property
    def alpha_mean(self) -> float:
        """Return the mean alpha-band PPC connectivity."""
        return self.mean_band_connectivity("alpha")

    @property
    def beta_mean(self) -> float:
        """Return the mean beta-band PPC connectivity."""
        return self.mean_band_connectivity("beta")

    @property
    def gamma_mean(self) -> float:
        """Return the mean gamma-band PPC connectivity."""
        return self.mean_band_connectivity("gamma")

    @property
    def full_mean(self) -> float:
        """Return the mean full-band PPC connectivity."""
        return self.mean_band_connectivity("full")


from dataclasses import dataclass

from features.config import FeatureExtractionConfig


@dataclass(frozen=True)
class PPCAnalysisEngineParametersFactory:
    """Factory used to build PPC analysis parameters from a global config."""

    @staticmethod
    def build_ppc_engine_parameters(
        config: FeatureExtractionConfig,
    ) -> SignalPPCAnalysisParameters:
        """Build PPC parameters from the global feature extraction config."""
        epoch_duration = config.ppc_epoch_duration
        epoch_overlap = config.ppc_epoch_overlap
        mode = config.ppc_mode
        n_jobs = config.ppc_n_jobs

        bands = tuple(
            SpectralBand(name=band_name, fmin=fmin, fmax=fmax)
            for band_name, (fmin, fmax) in config.bands.items()
        )

        return SignalPPCAnalysisParameters(
            bands=bands,
            epoch_duration=epoch_duration,
            epoch_overlap=epoch_overlap,
            mode=mode,
            n_jobs=n_jobs,
        )


import mne
from mne_connectivity import Connectivity, spectral_connectivity_epochs

from eeg.data import EEGProcessedData


class SignalPPCAnalysisEngine:
    """Engine used to compute PPC from an EEGProcessedData object."""

    def __init__(self, params: SignalPPCAnalysisParameters):
        self.params = params

    def _compute_epochs(self, raw: mne.io.Raw) -> mne.Epochs:
        """
        Split the continuous signal into fixed-length epochs.

        This is required by MNE-Connectivity for spectral connectivity
        computation.
        """
        epochs = mne.make_fixed_length_epochs(
            raw,
            duration=self.params.epoch_duration,
            overlap=self.params.epoch_overlap,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )

        if self.params.tmin is not None or self.params.tmax is not None:
            epochs = epochs.copy().crop(
                tmin=self.params.tmin,
                tmax=self.params.tmax,
            )

        return epochs

    def _compute_connectivity(self, epochs: mne.Epochs) -> Connectivity:
        """Compute PPC for all sensor pairs."""
        conn: Connectivity = spectral_connectivity_epochs(
            data=epochs,
            method="ppc",
            mode=self.params.mode,
            fmin=self.params.fmin,
            fmax=self.params.fmax,
            faverage=self.params.faverage,
            mt_adaptive=self.params.mt_adaptive,
            mt_low_bias=self.params.mt_low_bias,
            indices=None,
            block_size=self.params.block_size,
            n_jobs=self.params.n_jobs,
            verbose=False,
        )

        return conn

    def compute(self, eeg: EEGProcessedData) -> SignalPPCAnalysisResult:
        """Run the full PPC computation and return the result."""
        epochs = self._compute_epochs(eeg.raw)
        connectivity = self._compute_connectivity(epochs)

        result = SignalPPCAnalysisResult(
            eeg=eeg,
            params=self.params,
            connectivity=connectivity,
        )

        _ = result.dense_data

        return result


from dataclasses import dataclass


@dataclass(frozen=True)
class PPCVisualisationParameters:
    """Parameters used to configure PPC visualizations."""

    figsize_heatmap: tuple[float, float] = (10, 8)
    figsize_circle: tuple[float, float] = (10, 10)
    cmap_heatmap: str = "viridis"
    cmap_topomap: str = "magma"
    circle_facecolor: str = "black"
    circle_textcolor: str = "white"
    n_circle_lines: int = 40
    sensor_n_lines: int = 40
    threshold_quantile: float = 0.90
    show: bool = True


from functools import cached_property

import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle, plot_sensors_connectivity


class PPCVisualisationEngine:
    """
    Visualization engine for PPC results.

    Useful methods:
    - plot_band_heatmap(...)
    - plot_band_circle(...)
    - plot_band_sensor_graph(...)
    - plot_band_degree_topomap(...)
    - plot_summary(...)
    """

    def __init__(
        self,
        result: SignalPPCAnalysisResult,
        params: PPCVisualisationParameters | None = None,
    ):
        self.result = result
        self.params = params or PPCVisualisationParameters()

    @cached_property
    def info(self):
        """Return the MNE info object associated with the EEG data."""
        return self.result.eeg.info

    @cached_property
    def channel_names(self) -> list[str]:
        """Return the channel names used in the PPC result."""
        return self.result.channel_names

    def _get_band_matrix(self, band_name: str) -> np.ndarray:
        """Return the PPC matrix for a given frequency band."""
        return self.result.band_matrix(band_name)

    def _compute_node_strength(
        self,
        mat: np.ndarray,
        zero_diagonal: bool = True,
    ) -> np.ndarray:
        """Compute node strength as the sum of each electrode connectivity values."""
        work = mat.copy()

        if zero_diagonal:
            np.fill_diagonal(work, 0.0)

        return work.sum(axis=1)

    def _threshold_matrix(
        self,
        mat: np.ndarray,
        quantile: float | None = None,
    ) -> np.ndarray:
        """Keep only the strongest connections to simplify visualization."""
        quantile = self.params.threshold_quantile if quantile is None else quantile

        work = mat.copy()
        upper = work[np.triu_indices_from(work, k=1)]

        if len(upper) == 0:
            return work

        thr = np.quantile(upper, quantile)
        work[work < thr] = 0.0

        return work

    def plot_band_heatmap(self, band_name: str, mask_diagonal: bool = False):
        """Plot the full PPC matrix as a heatmap."""
        mat = self._get_band_matrix(band_name).copy()

        if mask_diagonal:
            np.fill_diagonal(mat, np.nan)

        is_symmetric = np.allclose(
            np.nan_to_num(mat, nan=0.0),
            np.nan_to_num(mat.T, nan=0.0),
            atol=1e-12,
        )

        fig, ax = plt.subplots(figsize=self.params.figsize_heatmap)
        im = ax.imshow(
            mat,
            cmap=self.params.cmap_heatmap,
            origin="lower",
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_title(f"PPC heatmap - {band_name} (symmetric={is_symmetric})")
        ax.set_xticks(np.arange(len(self.channel_names)))
        ax.set_yticks(np.arange(len(self.channel_names)))
        ax.set_xticklabels(self.channel_names, rotation=90)
        ax.set_yticklabels(self.channel_names)

        fig.colorbar(im, ax=ax, label="PPC")
        fig.tight_layout()

        if self.params.show:
            plt.show()

        return fig, ax

    def plot_band_circle(self, band_name: str, n_lines: int | None = None):
        """Plot the strongest connections as a circular graph."""
        mat = self._get_band_matrix(band_name)
        n_lines = self.params.n_circle_lines if n_lines is None else n_lines

        node_angles = circular_layout(
            node_names=self.channel_names,
            node_order=self.channel_names,
            start_pos=90,
            group_boundaries=None,
        )

        fig = plt.figure(figsize=self.params.figsize_circle)
        ax = fig.add_subplot(111, polar=True)

        plot_connectivity_circle(
            con=mat,
            node_names=self.channel_names,
            node_angles=node_angles,
            n_lines=n_lines,
            title=f"PPC circle - {band_name}",
            facecolor=self.params.circle_facecolor,
            textcolor=self.params.circle_textcolor,
            colorbar=True,
            show=False,
            ax=ax,
        )

        if self.params.show:
            plt.show()

        return fig, ax

    def plot_band_sensor_graph(
        self,
        band_name: str,
        n_lines: int | None = None,
        threshold_quantile: float | None = None,
    ):
        """Plot a 3D sensor graph showing the strongest connections."""
        mat = self._get_band_matrix(band_name)
        mat_thr = self._threshold_matrix(mat, quantile=threshold_quantile)
        n_lines = self.params.sensor_n_lines if n_lines is None else n_lines

        fig = plot_sensors_connectivity(
            info=self.info,
            con=mat_thr,
            n_con=n_lines,
            title=f"PPC sensor connectivity - {band_name}",
            verbose=False,
        )

        if self.params.show:
            plt.show()

        return fig

    def plot_band_degree_topomap(self, band_name: str, zero_diagonal: bool = True):
        """
        Plot a node-strength topomap.

        Node strength is defined as the sum of the connectivity values between
        each electrode and all other electrodes.
        """
        mat = self._get_band_matrix(band_name)
        node_strength = self._compute_node_strength(mat, zero_diagonal=zero_diagonal)

        fig, ax = plt.subplots(figsize=(7, 6))
        mne.viz.plot_topomap(
            node_strength,
            self.info,
            axes=ax,
            cmap=self.params.cmap_topomap,
            show=False,
            contours=6,
        )
        ax.set_title(f"PPC node strength topomap - {band_name}")

        if self.params.show:
            plt.show()

        return fig, ax

    def plot_band_summary(self, band_name: str):
        """Display the most useful visualizations for a given frequency band."""
        outputs = {}
        outputs["heatmap"] = self.plot_band_heatmap(band_name)
        outputs["circle"] = self.plot_band_circle(band_name)
        outputs["topomap"] = self.plot_band_degree_topomap(band_name)

        return outputs

    def plot_summary(self, bands: list[str] | None = None):
        """Display a visualization summary for several frequency bands."""
        bands = self.result.band_names if bands is None else bands
        outputs = {}

        for band_name in bands:
            outputs[band_name] = self.plot_band_summary(band_name)

        return outputs