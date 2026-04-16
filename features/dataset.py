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



from dataclasses import dataclass
from functools import cached_property
from typing import Any

import mne
import numpy as np
import pandas as pd

from participants.definition import ParticipantFactory



from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
import pandas as pd

from participants.definition import ParticipantFactory


def _canonical_edge_key(seed: str, target: str) -> str:
    a, b = sorted((str(seed), str(target)))
    return f"{a}__{b}"


@dataclass(slots=True)
class SingleParticipantProcessedFeatureDataset:
    """
    Dataset sujet-level après extraction complète.

    Notes perf
    ----------
    - `features_df` est supposé relativement petit par sujet, mais nombreux au total.
    - `ppc_band_results` doit idéalement contenir des matrices numpy float32.
    - Le cache est manuel car `cached_property` ne fonctionne pas avec `slots=True`
      sans `__dict__`.
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
        init=False, default=None, repr=False
    )
    _ppc_edge_keys_cache: list[str] | None = field(init=False, default=None, repr=False)
    _ppc_edge_dataframe_cache: pd.DataFrame | None = field(init=False, default=None, repr=False)

    @property
    def subject(self):
        if self._subject_cache is None:
            self._subject_cache = ParticipantFactory.build(self.subject_dico)
        return self._subject_cache

    @property
    def eeg_info(self):
        if self._eeg_info_cache is None:
            self._eeg_info_cache = mne.Info.from_json_dict(self.eeg_info_dico)
        return self._eeg_info_cache

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

    def ppc_matrix(self, band_name: str, dtype=np.float32) -> np.ndarray:
        if band_name not in self.ppc_band_results:
            raise KeyError(
                f"Unknown PPC band '{band_name}'. Available bands: {self.ppc_band_names}"
            )

        arr = self.ppc_band_results[band_name]
        if isinstance(arr, np.ndarray):
            return arr.astype(dtype, copy=False)

        return np.asarray(arr, dtype=dtype)

    @property
    def ppc_upper_triangle_indices(self) -> tuple[np.ndarray, np.ndarray]:
        if self._ppc_upper_triangle_indices_cache is None:
            n = len(self.ch_names)
            self._ppc_upper_triangle_indices_cache = np.triu_indices(n, k=1)
        return self._ppc_upper_triangle_indices_cache

    @property
    def ppc_edge_keys(self) -> list[str]:
        if self._ppc_edge_keys_cache is None:
            ch_names = self.ch_names
            ii, jj = self.ppc_upper_triangle_indices
            self._ppc_edge_keys_cache = [
                _canonical_edge_key(ch_names[i], ch_names[j])
                for i, j in zip(ii.tolist(), jj.tolist())
            ]
        return self._ppc_edge_keys_cache

    def to_psd_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.psd_band_results, orient="index")

    @property
    def ppc_edge_dataframe(self) -> pd.DataFrame:
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
                mat = self.ppc_matrix(band_name, dtype=np.float32)
                values = mat[ii, jj].astype(np.float32, copy=False)

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


from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split



from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


from features.results import FeatureExtractionResult, PSDBandExtractionResult, PPCBandExtractionResult


from typing import TYPE_CHECKING, Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split

from participants.groups import HealthState
from utils.enum import EnumParser

if TYPE_CHECKING:
    from features.dataset import FeaturesDataset
    from features.dataset import SingleParticipantProcessedFeatureDataset


class SampleSelector:
    """
    Helper orienté sélection / filtrage / projection ML pour manipuler
    un `FeaturesDataset`.

    Philosophie
    -----------
    - `FeaturesDataset` stocke et expose les vues tabulaires globales.
    - `SampleSelector` gère la logique métier de sélection, filtrage, split,
      et construction de X / y pour le machine learning.
    - Le selector évite autant que possible les recalculs et s'appuie sur
      les vues déjà cachées dans `FeaturesDataset`.

    Notes de performance
    --------------------
    - On exploite `dataset.wide_dataframe` et `dataset.subject_dataframe`
      déjà cachés côté dataset.
    - On évite les `copy()` tant qu'aucune modification n'est nécessaire.
    - Les colonnes ML sont déterminées par filtrage de noms, sans reconstruire
      les données sous-jacentes.
    """

    TARGET_COLUMN = "subject_health"

    SUBJECT_METADATA_COLUMNS = [
        "subject_id",
        "subject_tag",
        "subject_health",
        "subject_group",
        "subject_gender",
        "subject_mmse",
        "subject_age",
    ]

    CONNECTIVITY_PREFIX = "cn_"

    def __init__(self, dataset: "FeaturesDataset"):
        self.dataset = dataset

    # ==========================================================================
    # Descripteurs
    # ==========================================================================

    @property
    def scalar_feature_names(self) -> list[str]:
        """
        Noms des familles de features scalaires disponibles.
        """
        return list(self.dataset.scalar_feature_names)

    @property
    def connectivity_feature_names(self) -> list[str]:
        """
        Noms des familles de features de connectivité disponibles.
        Exemple :
        - cn_delta
        - cn_theta
        - cn_alpha
        - cn_beta
        """
        return list(self.dataset.connectivity_feature_names)

    @property
    def selectable_feature_names(self) -> list[str]:
        """
        Ensemble des noms de features acceptés par `.select(...)`.
        """
        return self.scalar_feature_names + self.connectivity_feature_names

    @property
    def n_subjects(self) -> int:
        return len(self.dataset.participant_datasets)

    @property
    def n_features(self) -> int:
        """
        Nombre de colonnes de X sans métadonnées.
        """
        return self.X().shape[1]

    # ==========================================================================
    # Helpers internes
    # ==========================================================================

    @staticmethod
    def _ensure_non_empty_list(values: Iterable[str], *, name: str) -> list[str]:
        """
        Convertit un itérable en liste et vérifie qu'il n'est pas vide.
        """
        result = list(values)
        if not result:
            raise ValueError(f"{name} cannot be empty.")
        return result

    def _clone_with_dataset(self, dataset: "FeaturesDataset") -> "SampleSelector":
        """
        Construit un nouveau selector sur un sous-dataset.
        """
        return SampleSelector(dataset)

    @classmethod
    def _is_connectivity_feature_name(cls, feature_name: str) -> bool:
        """
        Détermine si un nom de feature correspond à une famille de connectivité.
        """
        return str(feature_name).startswith(cls.CONNECTIVITY_PREFIX)

    @classmethod
    def _connectivity_feature_to_band_name(cls, feature_name: str) -> str:
        """
        Convertit :
        - cn_delta -> delta
        - cn_alpha -> alpha
        """
        if not cls._is_connectivity_feature_name(feature_name):
            raise ValueError(f"'{feature_name}' is not a connectivity feature name.")
        return feature_name[len(cls.CONNECTIVITY_PREFIX):]

    def _split_requested_features(
        self,
        features: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """
        Sépare les features demandées en deux groupes :
        - features scalaires
        - familles de connectivité
        """
        requested = self._ensure_non_empty_list(features, name="features")

        scalar_features: list[str] = []
        connectivity_features: list[str] = []

        for feature in requested:
            if self._is_connectivity_feature_name(feature):
                connectivity_features.append(feature)
            else:
                scalar_features.append(feature)

        return scalar_features, connectivity_features

    def _validate_requested_features(
        self,
        scalar_features: Sequence[str],
        connectivity_features: Sequence[str],
    ) -> None:
        """
        Vérifie que toutes les features demandées existent réellement.
        """
        missing_scalar = sorted(set(scalar_features) - set(self.scalar_feature_names))
        if missing_scalar:
            raise KeyError(
                f"Unknown scalar features: {missing_scalar}. "
                f"Available scalar features: {self.scalar_feature_names}"
            )

        missing_connectivity = sorted(
            set(connectivity_features) - set(self.connectivity_feature_names)
        )
        if missing_connectivity:
            raise KeyError(
                f"Unknown connectivity features: {missing_connectivity}. "
                f"Available connectivity features: {self.connectivity_feature_names}"
            )

    def _wide_feature_columns(
        self,
        *,
        include_scalar_features: Sequence[str] | None = None,
        include_connectivity_features: Sequence[str] | None = None,
    ) -> list[str]:
        """
        Détermine les colonnes de features à conserver dans la vue wide.

        Convention de nommage supposée
        ------------------------------
        - Features scalaires : <channel>_<feature>
          exemple : Fp1_entropy, Cz_theta_beta_ratio
        - Connectivité : cn_<band>_<seed>_<target>
          exemple : cn_delta_Fp1_Fp2

        Cette méthode ne reconstruit aucune donnée :
        elle filtre simplement les noms de colonnes déjà présents.
        """
        wide_columns = self.dataset.wide_dataframe.columns

        scalar_features = (
            list(include_scalar_features)
            if include_scalar_features is not None
            else list(self.scalar_feature_names)
        )
        connectivity_features = (
            list(include_connectivity_features)
            if include_connectivity_features is not None
            else list(self.connectivity_feature_names)
        )

        scalar_feature_columns: list[str] = []
        if scalar_features:
            scalar_suffixes = tuple(f"_{feature}" for feature in scalar_features)
            scalar_feature_columns = [
                col
                for col in wide_columns
                if col not in self.SUBJECT_METADATA_COLUMNS and col.endswith(scalar_suffixes)
            ]

        connectivity_feature_columns: list[str] = []
        if connectivity_features:
            connectivity_prefixes = tuple(f"{feature}_" for feature in connectivity_features)
            connectivity_feature_columns = [
                col
                for col in wide_columns
                if col not in self.SUBJECT_METADATA_COLUMNS and col.startswith(connectivity_prefixes)
            ]

        return scalar_feature_columns + connectivity_feature_columns

    # ==========================================================================
    # Filtrage / sélection
    # ==========================================================================

    def select(self, features: Sequence[str]) -> "FeaturesDataset":
        """
        Retourne un nouveau `FeaturesDataset` contenant uniquement les features
        demandées.

        `features` peut contenir un mélange de :
        - features scalaires : "theta_beta_ratio", "entropy", ...
        - connectivité      : "cn_delta", "cn_beta", ...

        Notes
        -----
        - Les métadonnées sujet sont conservées.
        - Les résultats PSD sont conservés tels quels.
        - Les bandes PPC inutiles sont supprimées.
        """
        from features.dataset import FeaturesDataset, SingleParticipantProcessedFeatureDataset

        scalar_features, connectivity_features = self._split_requested_features(features)
        self._validate_requested_features(scalar_features, connectivity_features)

        selected_band_names = {
            self._connectivity_feature_to_band_name(name)
            for name in connectivity_features
        }

        new_participant_datasets: list["SingleParticipantProcessedFeatureDataset"] = []

        for participant_dataset in self.dataset.participant_datasets:
            if scalar_features:
                new_features_df = participant_dataset.features_df.loc[:, scalar_features].copy()
            else:
                new_features_df = participant_dataset.features_df.iloc[:, 0:0].copy()

            if selected_band_names:
                new_ppc_band_results = {
                    band_name: value
                    for band_name, value in participant_dataset.ppc_band_results.items()
                    if band_name in selected_band_names
                }
            else:
                new_ppc_band_results = {}

            new_participant_datasets.append(
                SingleParticipantProcessedFeatureDataset(
                    features_df=new_features_df,
                    psd_band_results=participant_dataset.psd_band_results,
                    ppc_band_results=new_ppc_band_results,
                    subject_dico=participant_dataset.subject_dico,
                    pipeline_name=participant_dataset.pipeline_name,
                    eeg_info_dico=participant_dataset.eeg_info_dico,
                )
            )

        return FeaturesDataset(new_participant_datasets)

    def select_features(self, features: Sequence[str]) -> "SampleSelector":
        """
        Variante chaînable de `.select(...)`.
        """
        return self._clone_with_dataset(self.select(features))

    def drop(self, features: Sequence[str]) -> "FeaturesDataset":
        """
        Supprime certaines familles de features du dataset courant.
        """
        features_to_drop = set(self._ensure_non_empty_list(features, name="features"))

        unknown = sorted(features_to_drop - set(self.selectable_feature_names))
        if unknown:
            raise KeyError(
                f"Unknown features: {unknown}. "
                f"Available selectable features: {self.selectable_feature_names}"
            )

        kept_features = [
            feature_name
            for feature_name in self.selectable_feature_names
            if feature_name not in features_to_drop
        ]

        if not kept_features:
            raise ValueError("Dropping these features would leave no feature.")

        return self.select(kept_features)

    def drop_features(self, features: Sequence[str]) -> "SampleSelector":
        """
        Variante chaînable de `.drop(...)`.
        """
        return self._clone_with_dataset(self.drop(features))

    def select_subject_ids(self, subject_ids: Sequence[str]) -> "FeaturesDataset":
        """
        Retourne un nouveau `FeaturesDataset` limité aux sujets demandés.
        """
        from features.dataset import FeaturesDataset

        subject_ids_list = self._ensure_non_empty_list(subject_ids, name="subject_ids")
        subject_ids_set = set(subject_ids_list)

        new_participant_datasets = [
            participant_dataset
            for participant_dataset in self.dataset.participant_datasets
            if participant_dataset.subject.id in subject_ids_set
        ]

        if not new_participant_datasets:
            raise ValueError("No participant matched the requested subject_ids.")

        return FeaturesDataset(new_participant_datasets)

    def select_subject_ids_selector(self, subject_ids: Sequence[str]) -> "SampleSelector":
        """
        Variante chaînable de `.select_subject_ids(...)`.
        """
        return self._clone_with_dataset(self.select_subject_ids(subject_ids))

    def filter_by_healthstate(self, healthstates: Sequence["HealthState"]) -> "FeaturesDataset":
        """
        Filtre le dataset sur un ou plusieurs health states.
        """
        from features.dataset import FeaturesDataset

        parsed_healthstates = [
            EnumParser.parse(healthstate, HealthState).value
            for healthstate in healthstates
        ]

        new_participant_datasets = [
            participant_dataset
            for participant_dataset in self.dataset.participant_datasets
            if participant_dataset.subject.health_state in parsed_healthstates
        ]

        if not new_participant_datasets:
            raise ValueError(
                f"No participant matched the requested healthstates: {parsed_healthstates}"
            )

        return FeaturesDataset(new_participant_datasets)

    def filter_by_healthstate_selector(
        self,
        healthstates: Sequence["HealthState"],
    ) -> "SampleSelector":
        """
        Variante chaînable de `.filter_by_healthstate(...)`.
        """
        return self._clone_with_dataset(self.filter_by_healthstate(healthstates))

    
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import cached_property



from typing import Any

import pandas as pd



from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd

from utils.dataframe import DataframeHelpers


class FeaturesDataset:
    CONNECTIVITY_PREFIX = "cn_"
    EXCLUDED_CONNECTIVITY_BANDS = {"full"}

    SUBJECT_METADATA_COLUMNS = [
        "subject_id",
        #"subject_tag",
        "subject_health",
        "subject_group",
        "subject_gender",
        "subject_mmse",
        "subject_age",
    ]

    def __init__(self, participant_datasets: list["SingleParticipantProcessedFeatureDataset"]):
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
    def scalar_feature_names(self) -> list[str]:
        return self.participant_datasets[0].feature_names

    @property
    def psd_band_names(self) -> list[str]:
        return self.participant_datasets[0].psd_band_names

    @property
    def ppc_band_names(self) -> list[str]:
        return self.participant_datasets[0].ppc_band_names

    @property
    def connectivity_band_names(self) -> list[str]:
        return [
            band
            for band in self.ppc_band_names
            if band not in self.EXCLUDED_CONNECTIVITY_BANDS
        ]

    @property
    def connectivity_feature_names(self) -> list[str]:
        return [f"{self.CONNECTIVITY_PREFIX}{band}" for band in self.connectivity_band_names]

    @property
    def feature_names(self) -> list[str]:
        return self.scalar_feature_names + self.connectivity_feature_names

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

    @property
    def selector(self) -> "SampleSelector":
        return SampleSelector(self)
    
   

    def participant_dataset(self, participant_id: str) -> "SingleParticipantProcessedFeatureDataset":
        for dataset in self.participant_datasets:
            if dataset.subject.id == participant_id:
                return dataset
        raise KeyError(f"No participant dataset found for id='{participant_id}'.")

    @staticmethod
    def _edge_to_column_suffix(seed: str, target: str) -> str:
        return f"{seed}_{target}"

    @cached_property
    def _scalar_value_columns(self) -> list[str]:
        first = self.participant_datasets[0].features_df
        return [f"{ch}_{feat}" for ch in first.index for feat in first.columns]

    @cached_property
    def _connectivity_value_columns(self) -> list[str]:
        first = self.participant_datasets[0]
        ii, jj = first.ppc_upper_triangle_indices
        ch_names = first.ch_names

        cols = []
        for band in self.connectivity_band_names:
            for i, j in zip(ii.tolist(), jj.tolist()):
                cols.append(f"{self.CONNECTIVITY_PREFIX}{band}_{ch_names[i]}_{ch_names[j]}")
        return cols

    @cached_property
    def subject_dataframe(self) -> pd.DataFrame:
        rows = []
        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            rows.append(
                {
                    "subject_id": subject.id,
                    #"subject_tag": subject.tag,
                    "subject_health": subject.health_state,
                    "subject_group": subject.group,
                    "subject_gender": subject.gender,
                    "subject_age": subject.age,
                    "subject_mmse": subject.mmse,
                }
            )

        df = pd.DataFrame(rows)

        # Réduction mémoire utile pour colonnes répétitives
        for col in ["subject_tag", "subject_health", "subject_group", "subject_gender"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df

    @cached_property
    def wide_scalar_dataframe(self) -> pd.DataFrame:
        """
        Vue wide sujet-level des features scalaires uniquement.
        Construction vectorisée.
        """
        data = np.vstack([
            ds.features_df.to_numpy(dtype=np.float32, copy=False).ravel(order="C")
            for ds in self.participant_datasets
        ])

        values_df = pd.DataFrame(data, columns=self._scalar_value_columns)
        return pd.concat(
            [self.subject_dataframe.reset_index(drop=True), values_df.reset_index(drop=True)],
            axis=1,
        )

    @cached_property
    def wide_connectivity_dataframe(self) -> pd.DataFrame:
        """
        Vue wide sujet-level de la connectivité PPC.
        Construction vectorisée sur triangle supérieur.
        """
        first = self.participant_datasets[0]
        ii, jj = first.ppc_upper_triangle_indices

        rows = []
        for ds in self.participant_datasets:
            row_arrays = []
            for band in self.connectivity_band_names:
                mat = ds.ppc_matrix(band, dtype=np.float32)
                row_arrays.append(mat[ii, jj])
            if row_arrays:
                rows.append(np.concatenate(row_arrays, axis=0))
            else:
                rows.append(np.empty((0,), dtype=np.float32))

        data = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        values_df = pd.DataFrame(data, columns=self._connectivity_value_columns)

        return pd.concat(
            [self.subject_dataframe.reset_index(drop=True), values_df.reset_index(drop=True)],
            axis=1,
        )

    @cached_property
    def wide_dataframe(self) -> pd.DataFrame:
        """
        Vue wide complète.

        Important :
        - pas de merge
        - concat direct car l'ordre des sujets est identique
        """
        scalar_only = self.wide_scalar_dataframe
        conn_only = self.wide_connectivity_dataframe.drop(
            columns=self.SUBJECT_METADATA_COLUMNS,
            errors="ignore",
        )

        return pd.concat(
            [scalar_only.reset_index(drop=True), conn_only.reset_index(drop=True)],
            axis=1,
        )

    @cached_property
    def long_dataframe(self) -> pd.DataFrame:
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
            #df_long["subject_tag"] = subject.tag
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        df = pd.concat(rows, ignore_index=True)
        df["value"] = df["value"].astype(np.float32, copy=False)
        return df

    @cached_property
    def long_psd_dataframe(self) -> pd.DataFrame:
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
            #df_long["subject_tag"] = subject.tag
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        df = pd.concat(rows, ignore_index=True)
        df["value"] = df["value"].astype(np.float32, copy=False)
        return df

    @cached_property
    def long_ppc_dataframe(self) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            df_long = participant_dataset.ppc_edge_dataframe.copy()

            df_long["subject_id"] = subject.id
            #df_long["subject_tag"] = subject.tag
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    @cached_property
    def long_connectivity_dataframe(self) -> pd.DataFrame:
        df = self.long_ppc_dataframe
        df = df.loc[~df["band"].isin(self.EXCLUDED_CONNECTIVITY_BANDS)].copy()
        df["connectivity_feature"] = self.CONNECTIVITY_PREFIX + df["band"].astype(str)
        return df

    @cached_property
    def mean_feature_df(self) -> pd.DataFrame:
        return DataframeHelpers.mean(
            [dataset.features_df for dataset in self.participant_datasets]
        )

    @cached_property
    def mean_psd_df(self) -> pd.DataFrame:
        return DataframeHelpers.mean(
            [dataset.to_psd_dataframe() for dataset in self.participant_datasets]
        )

    @cached_property
    def all_feature_names(self):
        return list(self.wide_dataframe.columns)
    
    @cached_property
    def groups(self):
        return self.wide_dataframe["subject_id"]
    
    @cached_property
    def y(self):
        return self.wide_dataframe["subject_health"]

               
    
class SelectedFeaturesDataset(FeaturesDataset):
    def __init__(self, participant_datasets, selected_feature_names:list[str]):
        super().__init__(participant_datasets)
        self.selected_features = selected_feature_names

    @cached_property
    def X(self):
        return self.wide_dataframe[self.selected_features]
    

class FeaturesDatasetSelector:
    @staticmethod
    def select(dataset:FeaturesDataset, selection:list[str]):
        return SelectedFeaturesDataset(participant_datasets=dataset.participant_datasets, selected_feature_names=selection)

    
        

from features.factory import CompleteFeatureExtractionResult
from features.results import (
    FeatureExtractionResult,
    PSDBandExtractionResult,
    PPCBandExtractionResult,
)


class SingleParticipantProcessedFeatureDatasetFactory:
    @staticmethod
    def build(
        complete_extraction_result: CompleteFeatureExtractionResult,
    ) -> "SingleParticipantProcessedFeatureDataset":
        return SingleParticipantProcessedFeatureDataset(
            features_df=SingleParticipantProcessedFeatureDatasetFactory._build_features_df(
                complete_extraction_result.feature_result
            ),
            psd_band_results=SingleParticipantProcessedFeatureDatasetFactory._build_psd_dict(
                complete_extraction_result.psd_result
            ),
            ppc_band_results=SingleParticipantProcessedFeatureDatasetFactory._build_ppc_dict(
                complete_extraction_result.ppc_result
            ),
            subject_dico=dict(
                complete_extraction_result.feature_result.eeg.source.subject.to_dict()
            ),
            pipeline_name=str(complete_extraction_result.feature_result.eeg.pipeline_name),
            eeg_info_dico=dict(complete_extraction_result.feature_result.eeg_info_dico),
        )

    @staticmethod
    def _build_features_df(feature_result: FeatureExtractionResult) -> pd.DataFrame:
        """
        Convertit le résultat d'extraction des features scalaires
        en DataFrame [channels x features], en float32 pour réduire la mémoire.
        """
        df = feature_result.dataframe.copy()
        return df.astype(np.float32, copy=False)

    @staticmethod
    def _build_psd_dict(
        psd_result: PSDBandExtractionResult,
    ) -> dict[str, dict[str, float]]:
        """
        Les dicts JSON restent en float Python.
        """
        result: dict[str, dict[str, float]] = {}
        for signal_name, band_dict in psd_result.dico.items():
            result[signal_name] = {
                band_name: float(value)
                for band_name, value in band_dict.items()
            }
        return result

    @staticmethod
    def _build_ppc_dict(
        ppc_result: PPCBandExtractionResult,
    ) -> dict[str, np.ndarray]:
        """
        On garde des arrays numpy float32 en mémoire.
        C'est bien plus compact et plus rapide que des listes imbriquées.
        """
        return {
            band_name: np.asarray(ppc_result.matrix(band_name), dtype=np.float32)
            for band_name in ppc_result.band_names
        }