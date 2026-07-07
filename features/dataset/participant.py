from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
import pandas as pd

from eeg.data import EEGProcessedData
from participants.definition import ParticipantFactory


def _canonical_edge_key(seed: str, target: str) -> str:
    """
    Build a canonical undirected key for a connectivity edge.

    Example
    -------
    >>> _canonical_edge_key("Fp1", "Fp2")
    'Fp1__Fp2'

    Notes
    -----
    Channel order does not matter because the key is sorted.
    """
    a, b = sorted((str(seed), str(target)))

    return f"{a}__{b}"


@dataclass(slots=True)
class SingleParticipantProcessedFeatureDataset:
    """
    Subject-level dataset after complete EEG feature extraction.

    This container represents one participant after preprocessing and feature
    engineering.

    Parameters
    ----------
    features_df
        DataFrame with shape ``[channels x scalar_features]``.
    psd_band_results
        PSD dictionary indexed by channel and then by band:
        ``{channel_name: {band_name: value}}``.
    ppc_band_results
        Connectivity matrices indexed by band:
        ``{band_name: ndarray[n_channels, n_channels]}``.
    subject_dico
        Serialized subject representation.
    pipeline_name
        Name of the pipeline used to produce these data.
    eeg_info_dico
        Serialized representation of the ``mne.Info`` object.

    Notes
    -----
    - Caches are implemented manually because ``cached_property`` is not
      convenient with ``slots=True`` without ``__dict__``.
    - PPC matrices are ideally stored as ``float32`` to reduce memory usage.
    """

    features_df: pd.DataFrame
    psd_band_results: dict[str, dict[str, float]]
    ppc_band_results: dict[str, Any]
    subject_dico: dict[str, Any]
    pipeline_name: str
    eeg_info_dico: dict[str, Any]
    _eeg: EEGProcessedData

    _subject_cache: Any = field(init=False, default=None, repr=False)
    _eeg_info_cache: Any = field(init=False, default=None, repr=False)
    _ppc_upper_triangle_indices_cache: tuple[np.ndarray, np.ndarray] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ppc_edge_keys_cache: list[str] | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _ppc_edge_dataframe_cache: pd.DataFrame | None = field(
        init=False,
        default=None,
        repr=False,
    )

    @property
    def subject(self):
        """Return the subject object reconstructed from ``subject_dico``."""
        if self._subject_cache is None:
            self._subject_cache = ParticipantFactory.build(self.subject_dico)

        return self._subject_cache

    @property
    def eeg_info(self):
        """Return the ``mne.Info`` object reconstructed from ``eeg_info_dico``."""
        if self._eeg_info_cache is None:
            self._eeg_info_cache = mne.Info.from_json_dict(self.eeg_info_dico)

        return self._eeg_info_cache

    @property
    def feature_names(self) -> list[str]:
        """Return the scalar feature names available for each channel."""
        return list(self.features_df.columns)

    @property
    def ch_names(self) -> list[str]:
        """Return the EEG channel names."""
        return list(self.features_df.index)

    @property
    def psd_band_names(self) -> list[str]:
        """Return the available PSD band names."""
        if not self.psd_band_results:
            return []

        first_signal = next(iter(self.psd_band_results.values()))

        return list(first_signal.keys())

    @property
    def ppc_band_names(self) -> list[str]:
        """Return the available PPC connectivity band names."""
        return list(self.ppc_band_results.keys())

    def ppc_matrix(self, band_name: str, dtype=np.float32) -> np.ndarray:
        """
        Return the PPC matrix associated with a given band.

        Parameters
        ----------
        band_name
            Band name.
        dtype
            Desired NumPy dtype.

        Raises
        ------
        KeyError
            If the requested band does not exist.
        """
        if band_name not in self.ppc_band_results:
            raise KeyError(
                f"Unknown PPC band '{band_name}'. "
                f"Available bands: {self.ppc_band_names}"
            )

        arr = self.ppc_band_results[band_name]

        if isinstance(arr, np.ndarray):
            return arr.astype(dtype, copy=False)

        return np.asarray(arr, dtype=dtype)

    @property
    def ppc_upper_triangle_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return strict upper-triangular indices of PPC matrices.

        These indices are used to vectorize connectivity matrices without
        duplicating symmetric edges.
        """
        if self._ppc_upper_triangle_indices_cache is None:
            n_channels = len(self.ch_names)
            self._ppc_upper_triangle_indices_cache = np.triu_indices(n_channels, k=1)

        return self._ppc_upper_triangle_indices_cache

    @property
    def ppc_edge_keys(self) -> list[str]:
        """Return undirected edge keys associated with channel pairs."""
        if self._ppc_edge_keys_cache is None:
            ch_names = self.ch_names
            ii, jj = self.ppc_upper_triangle_indices

            self._ppc_edge_keys_cache = [
                _canonical_edge_key(ch_names[i], ch_names[j])
                for i, j in zip(ii.tolist(), jj.tolist())
            ]

        return self._ppc_edge_keys_cache

    def to_psd_dataframe(self) -> pd.DataFrame:
        """Convert PSD results to a channels-by-bands DataFrame."""
        return pd.DataFrame.from_dict(self.psd_band_results, orient="index")

    @property
    def ppc_edge_dataframe(self) -> pd.DataFrame:
        """
        Return the long-format view of PPC connectivity values.

        Columns
        -------
        - band
        - seed
        - target
        - edge
        - value
        """
        if self._ppc_edge_dataframe_cache is None:
            rows: list[pd.DataFrame] = []

            ch_names = self.ch_names
            ii, jj = self.ppc_upper_triangle_indices

            seed_arr = np.array([ch_names[i] for i in ii], dtype=object)
            target_arr = np.array([ch_names[j] for j in jj], dtype=object)
            edge_arr = np.array(
                [
                    _canonical_edge_key(ch_names[i], ch_names[j])
                    for i, j in zip(ii, jj)
                ],
                dtype=object,
            )

            for band_name in self.ppc_band_names:
                matrix = self.ppc_matrix(band_name, dtype=np.float32)
                values = matrix[ii, jj].astype(np.float32, copy=False)

                band_df = pd.DataFrame(
                    {
                        "band": band_name,
                        "seed": seed_arr,
                        "target": target_arr,
                        "edge": edge_arr,
                        "value": values,
                    }
                )
                rows.append(band_df)

            if not rows:
                self._ppc_edge_dataframe_cache = pd.DataFrame(
                    columns=["band", "seed", "target", "edge", "value"]
                )
            else:
                self._ppc_edge_dataframe_cache = pd.concat(rows, ignore_index=True)

        return self._ppc_edge_dataframe_cache

    @property
    def eeg(self):
        """Return the associated processed EEG object."""
        if self._eeg:
            return self._eeg

        raise ValueError("EEG does not exist")