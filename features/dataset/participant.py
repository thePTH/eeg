from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
import pandas as pd

from participants.definition import ParticipantFactory


def _canonical_edge_key(seed: str, target: str) -> str:
    """
    Construit une clé canonique non orientée pour une arête de connectivité.

    Exemple
    -------
    >>> _canonical_edge_key("Fp1", "Fp2")
    'Fp1__Fp2'

    Remarque
    --------
    L'ordre des canaux n'a pas d'importance : la clé est triée.
    """
    a, b = sorted((str(seed), str(target)))
    return f"{a}__{b}"


@dataclass(slots=True)
class SingleParticipantProcessedFeatureDataset:
    """
    Dataset sujet-level après extraction complète des features EEG.

    Ce conteneur représente les données d'un seul participant après passage
    dans le pipeline de preprocessing / feature engineering.

    Paramètres
    ----------
    features_df:
        DataFrame [channels x scalar_features].
    psd_band_results:
        Dictionnaire de PSD par canal puis bande :
        {channel_name: {band_name: value}}.
    ppc_band_results:
        Dictionnaire de matrices de connectivité par bande :
        {band_name: ndarray[n_channels, n_channels]}.
    subject_dico:
        Représentation sérialisée du sujet.
    pipeline_name:
        Nom du pipeline utilisé pour produire ces données.
    eeg_info_dico:
        Représentation sérialisée de l'objet `mne.Info`.

    Notes
    -----
    - Les caches sont manuels car `cached_property` est peu pratique avec
      `slots=True` sans `__dict__`.
    - Les matrices PPC sont idéalement stockées en `float32` pour limiter
      l'empreinte mémoire.
    """

    features_df: pd.DataFrame
    psd_band_results: dict[str, dict[str, float]]
    ppc_band_results: dict[str, Any]
    subject_dico: dict[str, Any]
    pipeline_name: str
    eeg_info_dico: dict[str, Any]

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
        """
        Objet sujet reconstruit à partir de `subject_dico`.
        """
        if self._subject_cache is None:
            self._subject_cache = ParticipantFactory.build(self.subject_dico)
        return self._subject_cache

    @property
    def eeg_info(self):
        """
        Objet `mne.Info` reconstruit à partir de `eeg_info_dico`.
        """
        if self._eeg_info_cache is None:
            self._eeg_info_cache = mne.Info.from_json_dict(self.eeg_info_dico)
        return self._eeg_info_cache

    @property
    def feature_names(self) -> list[str]:
        """
        Noms des features scalaires disponibles par canal.
        """
        return list(self.features_df.columns)

    @property
    def ch_names(self) -> list[str]:
        """
        Noms des canaux EEG.
        """
        return list(self.features_df.index)

    @property
    def psd_band_names(self) -> list[str]:
        """
        Noms des bandes PSD disponibles.
        """
        if not self.psd_band_results:
            return []

        first_signal = next(iter(self.psd_band_results.values()))
        return list(first_signal.keys())

    @property
    def ppc_band_names(self) -> list[str]:
        """
        Noms des bandes de connectivité disponibles.
        """
        return list(self.ppc_band_results.keys())

    def ppc_matrix(self, band_name: str, dtype=np.float32) -> np.ndarray:
        """
        Retourne la matrice PPC d'une bande donnée.

        Paramètres
        ----------
        band_name:
            Nom de la bande.
        dtype:
            Type numpy souhaité.

        Raises
        ------
        KeyError
            Si la bande demandée n'existe pas.
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
        Indices de la partie triangulaire supérieure stricte des matrices PPC.

        Utilisé pour vectoriser les matrices de connectivité sans dupliquer
        les arêtes symétriques.
        """
        if self._ppc_upper_triangle_indices_cache is None:
            n_channels = len(self.ch_names)
            self._ppc_upper_triangle_indices_cache = np.triu_indices(n_channels, k=1)

        return self._ppc_upper_triangle_indices_cache

    @property
    def ppc_edge_keys(self) -> list[str]:
        """
        Liste des clés d'arêtes non orientées associées aux couples de canaux.
        """
        if self._ppc_edge_keys_cache is None:
            ch_names = self.ch_names
            ii, jj = self.ppc_upper_triangle_indices

            self._ppc_edge_keys_cache = [
                _canonical_edge_key(ch_names[i], ch_names[j])
                for i, j in zip(ii.tolist(), jj.tolist())
            ]

        return self._ppc_edge_keys_cache

    def to_psd_dataframe(self) -> pd.DataFrame:
        """
        Convertit les résultats PSD en DataFrame [channels x bands].
        """
        return pd.DataFrame.from_dict(self.psd_band_results, orient="index")

    @property
    def ppc_edge_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des connectivités PPC.

        Colonnes
        --------
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
                [_canonical_edge_key(ch_names[i], ch_names[j]) for i, j in zip(ii, jj)],
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