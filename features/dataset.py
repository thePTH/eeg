from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mne
import numpy as np
import pandas as pd

from participants.definition import ParticipantFactory
from participants.groups import HealthState
from utils.dataframe import DataframeHelpers
from utils.enum import EnumParser


def _canonical_edge_key(seed: str, target: str) -> str:
    """
    Construit une clé canonique d'arête non orientée.

    Exemple
    -------
    >>> _canonical_edge_key("Fp2", "Fp1")
    'Fp1__Fp2'
    """
    a, b = sorted((str(seed), str(target)))
    return f"{a}__{b}"


@dataclass
class SingleParticipantProcessedFeatureDataset:
    """
    Dataset sujet-level après extraction complète.

    Paramètres
    ----------
    features_df:
        DataFrame principal des features scalaires, indexé par canal.
        shape attendue = (n_channels, n_features)

    psd_band_results:
        Résultats PSD agrégés par bande et par canal.
        Format attendu :
        {
            "Fp1": {"delta": ..., "theta": ..., ...},
            "Fp2": {...},
            ...
        }

    ppc_band_results:
        Résultats PPC agrégés par bande.
        Format attendu :
        {
            "delta": [[...], [...], ...],
            "theta": [[...], [...], ...],
            ...
        }
        ou bien directement des np.ndarray.
    """

    features_df: pd.DataFrame
    psd_band_results: dict[str, dict[str, float]]
    ppc_band_results: dict[str, Any]
    subject_dico: dict[str, Any]
    pipeline_name: str
    eeg_info_dico: dict[str, Any]

    def __post_init__(self):
        self.subject = ParticipantFactory.build(self.subject_dico)
        self.eeg_info = mne.Info.from_json_dict(self.eeg_info_dico)

    @property
    def feature_names(self) -> list[str]:
        return list(self.features_df.columns)

    @property
    def ch_names(self) -> list[str]:
        return list(self.features_df.index)

    @property
    def psd_band_names(self) -> list[str]:
        if not self.psd_band_results:
            return []
        first_signal = next(iter(self.psd_band_results.values()))
        return list(first_signal.keys())

    @property
    def ppc_band_names(self) -> list[str]:
        return list(self.ppc_band_results.keys())

    def ppc_matrix(self, band_name: str) -> np.ndarray:
        if band_name not in self.ppc_band_results:
            raise KeyError(
                f"Unknown PPC band '{band_name}'. Available bands: {self.ppc_band_names}"
            )
        return np.asarray(self.ppc_band_results[band_name], dtype=float)

    @property
    def ppc_edge_keys(self) -> list[str]:
        ch_names = self.ch_names
        keys: list[str] = []
        for i in range(len(ch_names)):
            for j in range(i + 1, len(ch_names)):
                keys.append(_canonical_edge_key(ch_names[i], ch_names[j]))
        return keys

    def to_psd_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.psd_band_results, orient="index")

    def to_ppc_edge_dataframe(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        ch_names = self.ch_names

        for band_name in self.ppc_band_names:
            mat = self.ppc_matrix(band_name)
            ii, jj = np.triu_indices_from(mat, k=1)
            for i, j in zip(ii.tolist(), jj.tolist()):
                seed = ch_names[i]
                target = ch_names[j]
                rows.append(
                    {
                        "band": band_name,
                        "seed": seed,
                        "target": target,
                        "edge": _canonical_edge_key(seed, target),
                        "value": float(mat[i, j]),
                    }
                )

        return pd.DataFrame(rows)


class SampleSelector:
    """
    Helper pratique pour filtrer rapidement le dataset global.
    """

    def __init__(self, dataset: "FeaturesDataset"):
        self.dataset = dataset
        self._long_features_df_cache: pd.DataFrame | None = None
        self._long_psd_df_cache: pd.DataFrame | None = None
        self._long_ppc_df_cache: pd.DataFrame | None = None

    @property
    def long_features_df(self) -> pd.DataFrame:
        if self._long_features_df_cache is None:
            self._long_features_df_cache = self.dataset.to_long_dataframe()
        return self._long_features_df_cache

    @property
    def long_psd_df(self) -> pd.DataFrame:
        if self._long_psd_df_cache is None:
            self._long_psd_df_cache = self.dataset.to_long_psd_dataframe()
        return self._long_psd_df_cache

    @property
    def long_ppc_df(self) -> pd.DataFrame:
        if self._long_ppc_df_cache is None:
            self._long_ppc_df_cache = self.dataset.to_long_ppc_dataframe()
        return self._long_ppc_df_cache


class FeaturesDataset:
    """
    Conteneur global regroupant tous les sujets d'une cohorte.

    Ce dataset expose trois vues longues distinctes :
    - features scalaires par canal
    - PSD par bande et par canal
    - PPC par bande et par arête

    C'est ce qu'il faut pour brancher les queries / bundles / engines
    de manière homogène.
    """

    def __init__(self, participant_datasets: list[SingleParticipantProcessedFeatureDataset]):
        if not participant_datasets:
            raise ValueError("participant_datasets cannot be empty.")
        self.participant_datasets = participant_datasets

    @property
    def subjects(self):
        return [dataset.subject for dataset in self.participant_datasets]

    @property
    def ch_names(self) -> list[str]:
        return self.participant_datasets[0].ch_names

    @property
    def feature_names(self) -> list[str]:
        return self.participant_datasets[0].feature_names

    @property
    def psd_band_names(self) -> list[str]:
        return self.participant_datasets[0].psd_band_names

    @property
    def ppc_band_names(self) -> list[str]:
        return self.participant_datasets[0].ppc_band_names

    @property
    def ppc_edge_keys(self) -> list[str]:
        return self.participant_datasets[0].ppc_edge_keys

    @property
    def groups(self):
        return set(subject.group for subject in self.subjects)

    @property
    def eeg_info(self):
        return self.participant_datasets[0].eeg_info

    @property
    def pipeline_name(self) -> str:
        return self.participant_datasets[0].pipeline_name

    def participant_dataset(self, participant_id: str) -> SingleParticipantProcessedFeatureDataset:
        for dataset in self.participant_datasets:
            if dataset.subject.id == participant_id:
                return dataset
        raise KeyError(f"No participant dataset found for id='{participant_id}'.")

    def filter_by_healthstate(self, healthstate: HealthState):
        healthstate = EnumParser.parse(healthstate, HealthState).value
        return FeaturesDataset(
            [
                dataset
                for dataset in self.participant_datasets
                if dataset.subject.health_state == healthstate
            ]
        )

    # Alias pour compatibilité avec l'ancien typo
    def filter_by_healthsate(self, healthstate: HealthState):
        return self.filter_by_healthstate(healthstate)

    def to_long_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des features scalaires.

        Colonnes retournées
        ------------------
        subject_id, subject_age, subject_mmse, subject_health,
        channel, feature, value
        """
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            features_df = participant_dataset.features_df

            df_long = (
                features_df
                .reset_index(names="channel")
                .melt(id_vars="channel", var_name="feature", value_name="value")
            )

            df_long["subject_id"] = subject.id
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_long_psd_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des résultats PSD agrégés par bande.

        Colonnes retournées
        ------------------
        subject_id, subject_age, subject_mmse, subject_health,
        channel, band, value
        """
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            psd_df = participant_dataset.to_psd_dataframe()

            df_long = (
                psd_df
                .reset_index(names="channel")
                .melt(id_vars="channel", var_name="band", value_name="value")
            )

            df_long["subject_id"] = subject.id
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_long_ppc_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des résultats PPC par bande et par arête.

        Colonnes retournées
        ------------------
        subject_id, subject_age, subject_mmse, subject_health,
        band, seed, target, edge, value
        """
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            df_long = participant_dataset.to_ppc_edge_dataframe().copy()

            df_long["subject_id"] = subject.id
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_wide_dataframe(self) -> pd.DataFrame:
        rows = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            features_df = participant_dataset.features_df

            row = {
                "subject_id": subject.id,
                "subject_health": subject.health_state,
                "subject_mmse": subject.mmse,
                "subject_age": subject.age,
            }

            for channel in features_df.index:
                for feature in features_df.columns:
                    row[f"{channel}_{feature}"] = float(features_df.loc[channel, feature])

            rows.append(row)

        return pd.DataFrame(rows)

    def to_subject_dataframe(self) -> pd.DataFrame:
        rows = []
        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            rows.append(
                {
                    "subject_id": subject.id,
                    "subject_health": subject.health_state,
                    "subject_age": subject.age,
                    "subject_mmse": subject.mmse,
                }
            )
        return pd.DataFrame(rows)

    @property
    def mean_feature_df(self) -> pd.DataFrame:
        return DataframeHelpers.mean(
            [dataset.features_df for dataset in self.participant_datasets]
        )

    @property
    def mean_psd_df(self) -> pd.DataFrame:
        return DataframeHelpers.mean(
            [dataset.to_psd_dataframe() for dataset in self.participant_datasets]
        )

    @property
    def selector(self) -> SampleSelector:
        return SampleSelector(self)
    




from dataclasses import asdict
from typing import Any

import pandas as pd

from features.results import (
    FeatureExtractionResult,
    PSDBandExtractionResult,
    PPCBandExtractionResult,
)


class SingleParticipantProcessedFeatureDatasetFactory:
    """
    Factory pour construire un SingleParticipantProcessedFeatureDataset
    à partir des objets de résultats d'extraction.

    Cette factory centralise :
    - la conversion des résultats features -> DataFrame
    - la conversion des résultats PSD -> dict sérialisable
    - la conversion des résultats PPC -> dict sérialisable
    - l'emballage des métadonnées sujet / pipeline / EEG info

    Remarque
    --------
    On stocke :
    - les features sous forme de DataFrame
    - la PSD et la PPC sous forme de dict
      pour simplifier l'export / import du dataset
    """

    @staticmethod
    def build(
        *,
        feature_result: FeatureExtractionResult,
        psd_result: PSDBandExtractionResult,
        ppc_result: PPCBandExtractionResult,
        subject_dico: dict[str, Any],
        pipeline_name: str,
    ) -> "SingleParticipantProcessedFeatureDataset":
        return SingleParticipantProcessedFeatureDataset(
            features_df=SingleParticipantProcessedFeatureDatasetFactory._build_features_df(
                feature_result
            ),
            psd_band_results=SingleParticipantProcessedFeatureDatasetFactory._build_psd_dict(
                psd_result
            ),
            ppc_band_results=SingleParticipantProcessedFeatureDatasetFactory._build_ppc_dict(
                ppc_result
            ),
            subject_dico=dict(subject_dico),
            pipeline_name=str(pipeline_name),
            eeg_info_dico=feature_result.eeg.info.to_json_dict(),
        )

    @staticmethod
    def _build_features_df(feature_result: FeatureExtractionResult) -> pd.DataFrame:
        """
        Convertit le résultat d'extraction des features scalaires
        en DataFrame [channels x features].
        """
        df = feature_result.dataframe.copy()

        # On force les float pour éviter les surprises à l'export / import
        for col in df.columns:
            df[col] = df[col].astype(float)

        return df

    @staticmethod
    def _build_psd_dict(psd_result: PSDBandExtractionResult) -> dict[str, dict[str, float]]:
        """
        Convertit le résultat PSD en dict sérialisable :
        {
            "Fp1": {"delta": ..., "theta": ..., ...},
            ...
        }
        """
        return {
            signal_name: {
                band_name: float(value)
                for band_name, value in band_dict.items()
            }
            for signal_name, band_dict in psd_result.dico.items()
        }

    @staticmethod
    def _build_ppc_dict(ppc_result: PPCBandExtractionResult) -> dict[str, list[list[float]]]:
        """
        Convertit le résultat PPC en dict sérialisable :
        {
            "delta": [[...], [...], ...],
            "theta": [[...], [...], ...],
            ...
        }
        """
        return ppc_result.to_serializable_dict()