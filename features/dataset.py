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



from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split



from typing import Iterable, Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


from features.results import FeatureExtractionResult, PSDBandExtractionResult, PPCBandExtractionResult

class SampleSelector:
    """
    Helper orienté sélection / filtrage / projection ML pour manipuler
    un `FeaturesDataset`.

    Responsabilités
    ---------------
    Cette classe gère :
    - la sélection de features
    - le filtrage de sujets
    - le filtrage métier (ex. health state)
    - les splits train / test
    - la construction de X / y pour sklearn

    Elle ne gère pas la logique d'affichage / représentation tabulaire :
    celle-ci reste dans `FeaturesDataset`.
    """

    TARGET_COLUMN = "subject_health"

    SUBJECT_METADATA_COLUMNS = [
        "subject_id",
        "subject_health",
        "subject_group",
        "subject_gender",
        "subject_mmse",
        "subject_age",
    ]

    CONNECTIVITY_PREFIX = "cn_"
    EXCLUDED_CONNECTIVITY_FEATURES = {"cn_full"}

    def __init__(self, dataset: "FeaturesDataset"):
        self.dataset = dataset

    # ==========================================================================
    # Descripteurs
    # ==========================================================================

    @property
    def scalar_feature_names(self) -> list[str]:
        """
        Noms des features scalaires disponibles.
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
        Ensemble des noms acceptés par `.select(...)`.
        """
        return self.scalar_feature_names + self.connectivity_feature_names

    @property
    def n_subjects(self) -> int:
        return len(self.dataset.participant_datasets)

    # ==========================================================================
    # Helpers internes
    # ==========================================================================

    @staticmethod
    def _ensure_non_empty_list(values: Iterable[str], *, name: str) -> list[str]:
        result = list(values)
        if not result:
            raise ValueError(f"{name} cannot be empty.")
        return result

    def _clone_with_dataset(self, dataset: "FeaturesDataset") -> "SampleSelector":
        return SampleSelector(dataset)

    @classmethod
    def _is_connectivity_feature_name(cls, feature_name: str) -> bool:
        return feature_name.startswith(cls.CONNECTIVITY_PREFIX)

    def _split_requested_features(
        self,
        features: Sequence[str],
    ) -> tuple[list[str], list[str]]:
        """
        Sépare les features demandées en :
        - features scalaires
        - features de connectivité
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

    @classmethod
    def _connectivity_feature_to_band_name(cls, feature_name: str) -> str:
        """
        Convertit 'cn_delta' -> 'delta'
        """
        if not cls._is_connectivity_feature_name(feature_name):
            raise ValueError(f"'{feature_name}' is not a connectivity feature name.")
        return feature_name[len(cls.CONNECTIVITY_PREFIX):]

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
        Calcule les colonnes wide à conserver dans X.
        """
        wide_columns = list(self.dataset.to_wide_dataframe().columns)

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
        scalar_suffixes = tuple(f"_{feature}" for feature in scalar_features)

        connectivity_feature_columns: list[str] = []
        connectivity_prefixes = tuple(f"{feature}_" for feature in connectivity_features)

        for col in wide_columns:
            if col in self.SUBJECT_METADATA_COLUMNS:
                continue

            if scalar_suffixes and col.endswith(scalar_suffixes):
                scalar_feature_columns.append(col)
                continue

            if connectivity_prefixes and col.startswith(connectivity_prefixes):
                connectivity_feature_columns.append(col)
                continue

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
        """
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
        Supprime certaines features du dataset courant.
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

    # ==========================================================================
    # Split train / test
    # ==========================================================================

    def split_train_test(
        self,
        *,
        test_size: float = 0.2,
        random_state: int | None = 42,
        stratify: bool = True,
        shuffle: bool = True,
    ) -> tuple["FeaturesDataset", "FeaturesDataset"]:
        """
        Découpe le dataset en train / test au niveau sujet.
        """
        subject_df = self.dataset.to_subject_dataframe().copy()

        if len(subject_df) < 2:
            raise ValueError("At least 2 subjects are required to split the dataset.")

        stratify_values = subject_df[self.TARGET_COLUMN] if stratify else None

        train_ids, test_ids = train_test_split(
            subject_df["subject_id"],
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=stratify_values,
        )

        train_dataset = self.select_subject_ids(train_ids.tolist())
        test_dataset = self.select_subject_ids(test_ids.tolist())

        return train_dataset, test_dataset

    def split_train_test_selectors(
        self,
        *,
        test_size: float = 0.2,
        random_state: int | None = 42,
        stratify: bool = True,
        shuffle: bool = True,
    ) -> tuple["SampleSelector", "SampleSelector"]:
        """
        Variante renvoyant directement deux `SampleSelector`.
        """
        train_dataset, test_dataset = self.split_train_test(
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )
        return SampleSelector(train_dataset), SampleSelector(test_dataset)

    def train_test_Xy(
        self,
        *,
        test_size: float = 0.2,
        random_state: int | None = 42,
        stratify: bool = True,
        shuffle: bool = True,
        include_metadata: bool = False,
        metadata_columns: Sequence[str] | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Retourne directement :
        X_train, X_test, y_train, y_test
        """
        train_selector, test_selector = self.split_train_test_selectors(
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
            shuffle=shuffle,
        )

        X_train = train_selector.X(
            include_metadata=include_metadata,
            metadata_columns=metadata_columns,
            drop_columns=drop_columns,
        )
        y_train = train_selector.y()

        X_test = test_selector.X(
            include_metadata=include_metadata,
            metadata_columns=metadata_columns,
            drop_columns=drop_columns,
        )
        y_test = test_selector.y()

        return X_train, X_test, y_train, y_test

    # ==========================================================================
    # Projection ML
    # ==========================================================================

    def X(
        self,
        *,
        include_metadata: bool = False,
        metadata_columns: Sequence[str] | None = None,
        include_scalar_features: Sequence[str] | None = None,
        include_connectivity_features: Sequence[str] | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Construit la matrice X compatible scikit-learn.
        """
        df = self.dataset.to_wide_dataframe().copy()

        if include_scalar_features is not None:
            unknown_scalar = sorted(set(include_scalar_features) - set(self.scalar_feature_names))
            if unknown_scalar:
                raise KeyError(
                    f"Unknown scalar features: {unknown_scalar}. "
                    f"Available scalar features: {self.scalar_feature_names}"
                )

        if include_connectivity_features is not None:
            unknown_connectivity = sorted(
                set(include_connectivity_features) - set(self.connectivity_feature_names)
            )
            if unknown_connectivity:
                raise KeyError(
                    f"Unknown connectivity features: {unknown_connectivity}. "
                    f"Available connectivity features: {self.connectivity_feature_names}"
                )

        feature_columns = self._wide_feature_columns(
            include_scalar_features=include_scalar_features,
            include_connectivity_features=include_connectivity_features,
        )

        X = df.loc[:, feature_columns].copy()

        if include_metadata:
            if metadata_columns is None:
                kept_metadata = [
                    col for col in self.SUBJECT_METADATA_COLUMNS
                    if col != self.TARGET_COLUMN and col in df.columns
                ]
            else:
                kept_metadata = [
                    col for col in metadata_columns
                    if col != self.TARGET_COLUMN and col in df.columns
                ]

            if kept_metadata:
                X = pd.concat([df.loc[:, kept_metadata].copy(), X], axis=1)

        if drop_columns is not None:
            X = X.drop(columns=list(drop_columns), errors="ignore")

        return X

    def y(self) -> pd.Series:
        """
        Retourne la cible à prédire.
        """
        df = self.dataset.to_wide_dataframe()
        if self.TARGET_COLUMN not in df.columns:
            raise KeyError(
                f"Target column '{self.TARGET_COLUMN}' not found in wide dataframe."
            )
        return df[self.TARGET_COLUMN].copy()

    def Xy(
        self,
        *,
        include_metadata: bool = False,
        metadata_columns: Sequence[str] | None = None,
        include_scalar_features: Sequence[str] | None = None,
        include_connectivity_features: Sequence[str] | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Retourne directement `(X, y)`.
        """
        X = self.X(
            include_metadata=include_metadata,
            metadata_columns=metadata_columns,
            include_scalar_features=include_scalar_features,
            include_connectivity_features=include_connectivity_features,
            drop_columns=drop_columns,
        )
        y = self.y()
        return X, y

    def feature_matrix(
        self,
        *,
        include_metadata: bool = False,
        metadata_columns: Sequence[str] | None = None,
        include_scalar_features: Sequence[str] | None = None,
        include_connectivity_features: Sequence[str] | None = None,
        drop_columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        return self.X(
            include_metadata=include_metadata,
            metadata_columns=metadata_columns,
            include_scalar_features=include_scalar_features,
            include_connectivity_features=include_connectivity_features,
            drop_columns=drop_columns,
        )

    def target(self) -> pd.Series:
        return self.y()

    def y_encoded(self, mapping: dict[str, int] | None = None) -> pd.Series:
        """
        Retourne une version encodée numériquement de la cible.
        """
        y = self.y()

        if mapping is None:
            classes = sorted(y.dropna().unique().tolist())
            mapping = {label: i for i, label in enumerate(classes)}

        encoded = y.map(mapping)

        if encoded.isna().any():
            unknown_labels = sorted(y[encoded.isna()].unique().tolist())
            raise ValueError(
                f"Some labels could not be encoded with mapping={mapping}. "
                f"Unknown labels: {unknown_labels}"
            )

        return encoded

    # ==========================================================================
    # Inspection / validation
    # ==========================================================================

    @property
    def n_features(self) -> int:
        return self.X().shape[1]

    def class_distribution(self) -> pd.Series:
        return self.y().value_counts(dropna=False)

    def summary(self) -> dict[str, object]:
        X = self.X()
        y = self.y()

        return {
            "n_subjects": len(y),
            "n_features": X.shape[1],
            "n_scalar_feature_families": len(self.scalar_feature_names),
            "n_connectivity_feature_families": len(self.connectivity_feature_names),
            "target_name": self.TARGET_COLUMN,
            "classes": sorted(y.dropna().unique().tolist()),
            "class_distribution": y.value_counts(dropna=False).to_dict(),
        }

    def assert_ml_ready(self) -> None:
        X, y = self.Xy()

        if X.empty:
            raise ValueError("X is empty.")

        if y.empty:
            raise ValueError("y is empty.")

        if y.nunique(dropna=True) < 2:
            raise ValueError("At least two target classes are required.")

        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(
                f"X contains NaN values. Problematic columns include: {nan_cols[:10]}"
            )

        if y.isna().any():
            raise ValueError("y contains NaN values.")


from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



from typing import Any

import pandas as pd


class FeaturesDataset:
    """
    Conteneur global regroupant tous les sujets d'une cohorte.

    Responsabilités
    ---------------
    Cette classe :
    - stocke les participant datasets
    - expose des vues tabulaires / représentations des données
    - expose quelques métadonnées globales

    Elle ne gère pas la logique de filtrage / split / sélection métier :
    cela est délégué à `SampleSelector`.
    """

    CONNECTIVITY_PREFIX = "cn_"
    EXCLUDED_CONNECTIVITY_BANDS = {"full"}

    SUBJECT_METADATA_COLUMNS = [
        "subject_id",
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

    # -------------------------------------------------------------------------
    # Métadonnées globales
    # -------------------------------------------------------------------------

    @property
    def subjects(self):
        return [dataset.subject for dataset in self.participant_datasets]

    @property
    def ch_names(self) -> list[str]:
        return self.participant_datasets[0].ch_names

    @property
    def scalar_feature_names(self) -> list[str]:
        """
        Noms des features scalaires natives présentes dans `features_df`.
        """
        return self.participant_datasets[0].feature_names

    @property
    def psd_band_names(self) -> list[str]:
        return self.participant_datasets[0].psd_band_names

    @property
    def ppc_band_names(self) -> list[str]:
        return self.participant_datasets[0].ppc_band_names

    @property
    def connectivity_band_names(self) -> list[str]:
        """
        Bandes PPC exposées comme familles de connectivité.
        La bande 'full' est exclue.
        """
        return [
            band
            for band in self.ppc_band_names
            if band not in self.EXCLUDED_CONNECTIVITY_BANDS
        ]

    @property
    def connectivity_feature_names(self) -> list[str]:
        """
        Noms des pseudo-features de connectivité :
        - cn_delta
        - cn_theta
        - cn_alpha
        - ...
        """
        return [
            f"{self.CONNECTIVITY_PREFIX}{band}"
            for band in self.connectivity_band_names
        ]

    @property
    def feature_names(self) -> list[str]:
        """
        Ensemble des features de haut niveau sélectionnables :
        - features scalaires
        - familles de connectivité
        """
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

    # -------------------------------------------------------------------------
    # Accès participant
    # -------------------------------------------------------------------------

    def participant_dataset(self, participant_id: str) -> "SingleParticipantProcessedFeatureDataset":
        for dataset in self.participant_datasets:
            if dataset.subject.id == participant_id:
                return dataset
        raise KeyError(f"No participant dataset found for id='{participant_id}'.")

    # -------------------------------------------------------------------------
    # Vues longues
    # -------------------------------------------------------------------------

    def to_long_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des features scalaires.

        Colonnes :
        subject_id, subject_age, subject_mmse, subject_health,
        subject_group, subject_gender,
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
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_long_psd_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des résultats PSD agrégés par bande.

        Colonnes :
        subject_id, subject_age, subject_mmse, subject_health,
        subject_group, subject_gender,
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
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_long_ppc_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des résultats PPC par bande et par arête.

        Colonnes :
        subject_id, subject_age, subject_mmse, subject_health,
        subject_group, subject_gender,
        band, seed, target, edge, value
        """
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            df_long = participant_dataset.to_ppc_edge_dataframe().copy()

            df_long["subject_id"] = subject.id
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    def to_long_connectivity_dataframe(self) -> pd.DataFrame:
        """
        Vue longue de la connectivité avec convention de nommage orientée
        feature de haut niveau.

        Colonnes :
        subject_id, subject_age, subject_mmse, subject_health,
        subject_group, subject_gender,
        connectivity_feature, band, seed, target, edge, value
        """
        df = self.to_long_ppc_dataframe().copy()
        df = df.loc[~df["band"].isin(self.EXCLUDED_CONNECTIVITY_BANDS)].copy()
        df["connectivity_feature"] = self.CONNECTIVITY_PREFIX + df["band"].astype(str)
        return df

    # -------------------------------------------------------------------------
    # Vues wide
    # -------------------------------------------------------------------------

    @staticmethod
    def _edge_to_column_suffix(seed: str, target: str) -> str:
        """
        Convertit une arête en suffixe stable de colonne.

        Exemple :
        Fp1, Fp2 -> Fp1_Fp2
        """
        return f"{seed}_{target}"

    def to_wide_scalar_dataframe(self) -> pd.DataFrame:
        """
        Vue wide sujet-level des features scalaires uniquement.

        Colonnes :
        - métadonnées sujet
        - <channel>_<feature>
        """
        rows = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            features_df = participant_dataset.features_df

            row = {
                "subject_id": subject.id,
                "subject_health": subject.health_state,
                "subject_group": subject.group,
                "subject_gender": subject.gender,
                "subject_mmse": subject.mmse,
                "subject_age": subject.age,
            }

            for channel in features_df.index:
                for feature in features_df.columns:
                    row[f"{channel}_{feature}"] = float(features_df.loc[channel, feature])

            rows.append(row)

        return pd.DataFrame(rows)

    def to_wide_connectivity_dataframe(self) -> pd.DataFrame:
        """
        Vue wide sujet-level de la connectivité PPC.

        Colonnes :
        - métadonnées sujet
        - cn_<band>_<seed>_<target>

        Exemple :
        - cn_delta_Fp1_Fp2
        - cn_alpha_C3_Pz

        La bande 'full' est exclue.
        """
        rows = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject

            row = {
                "subject_id": subject.id,
                "subject_health": subject.health_state,
                "subject_group": subject.group,
                "subject_gender": subject.gender,
                "subject_mmse": subject.mmse,
                "subject_age": subject.age,
            }

            df_ppc = participant_dataset.to_ppc_edge_dataframe()

            if not df_ppc.empty:
                df_ppc = df_ppc.loc[
                    ~df_ppc["band"].isin(self.EXCLUDED_CONNECTIVITY_BANDS)
                ].copy()

                for record in df_ppc.itertuples(index=False):
                    col_name = (
                        f"{self.CONNECTIVITY_PREFIX}"
                        f"{record.band}_"
                        f"{self._edge_to_column_suffix(record.seed, record.target)}"
                    )
                    row[col_name] = float(record.value)

            rows.append(row)

        return pd.DataFrame(rows)

    def to_wide_dataframe(self) -> pd.DataFrame:
        """
        Vue wide sujet-level complète.

        Colonnes :
        - métadonnées sujet
        - features scalaires      : <channel>_<feature>
        - connectivité PPC        : cn_<band>_<seed>_<target>
        """
        scalar_df = self.to_wide_scalar_dataframe()
        connectivity_df = self.to_wide_connectivity_dataframe()

        return scalar_df.merge(
            connectivity_df,
            on=self.SUBJECT_METADATA_COLUMNS,
            how="outer",
        )

    # -------------------------------------------------------------------------
    # Vue sujet-level
    # -------------------------------------------------------------------------

    def to_subject_dataframe(self) -> pd.DataFrame:
        """
        Vue sujet-level minimale.
        """
        rows = []
        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            rows.append(
                {
                    "subject_id": subject.id,
                    "subject_health": subject.health_state,
                    "subject_group": subject.group,
                    "subject_gender": subject.gender,
                    "subject_age": subject.age,
                    "subject_mmse": subject.mmse,
                }
            )
        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Résumés agrégés
    # -------------------------------------------------------------------------

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

