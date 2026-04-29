from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

from sklearn.model_selection import train_test_split

from participants.groups import HealthState
from utils.enum import EnumParser

if TYPE_CHECKING:
    from .base import FeaturesDataset
    from .participant import SingleParticipantProcessedFeatureDataset
    from .selected import SelectedFeaturesDataset


class SampleSelector:
    """
    Helper orienté sélection, filtrage et découpage d'un `FeaturesDataset`.

    Cette classe manipule les familles métier :
    - entropy
    - variance
    - cn_alpha
    - subject_age
    etc.
    """

    TARGET_COLUMN = "subject_health"

    CONNECTIVITY_PREFIX = "cn_"
    SUBJECT_FEATURE_PREFIX = "subject_"

    def __init__(self, dataset: "FeaturesDataset"):
        self.dataset = dataset

    # ==========================================================================
    # Descripteurs simples
    # ==========================================================================

    @property
    def scalar_feature_names(self) -> list[str]:
        return list(self.dataset.scalar_feature_names)

    @property
    def connectivity_feature_names(self) -> list[str]:
        return list(self.dataset.connectivity_feature_names)

    @property
    def subject_feature_names(self) -> list[str]:
        return list(self.dataset.subject_feature_names)

    @property
    def selectable_feature_names(self) -> list[str]:
        return (
            self.scalar_feature_names
            + self.connectivity_feature_names
            + self.subject_feature_names
        )

    @property
    def n_subjects(self) -> int:
        return len(self.dataset.participant_datasets)

    @property
    def n_features(self) -> int:
        return self.X().shape[1]

    def X(self):
        return self.dataset.X

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
        return str(feature_name).startswith(cls.CONNECTIVITY_PREFIX)

    @classmethod
    def _is_subject_feature_name(cls, feature_name: str) -> bool:
        return str(feature_name).startswith(cls.SUBJECT_FEATURE_PREFIX)

    @classmethod
    def _connectivity_feature_to_band_name(cls, feature_name: str) -> str:
        if not cls._is_connectivity_feature_name(feature_name):
            raise ValueError(f"'{feature_name}' is not a connectivity feature name.")

        return feature_name[len(cls.CONNECTIVITY_PREFIX):]

    def _split_requested_features(
        self,
        features: Sequence[str],
    ) -> tuple[list[str], list[str], list[str]]:
        requested = self._ensure_non_empty_list(features, name="features")

        scalar_features = []
        connectivity_features = []
        subject_features = []

        for feature in requested:
            if self._is_connectivity_feature_name(feature):
                connectivity_features.append(feature)

            elif self._is_subject_feature_name(feature):
                subject_features.append(feature)

            else:
                scalar_features.append(feature)

        return scalar_features, connectivity_features, subject_features

    def _validate_requested_features(
        self,
        scalar_features: Sequence[str],
        connectivity_features: Sequence[str],
        subject_features: Sequence[str],
    ) -> None:

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

        missing_subject = sorted(set(subject_features) - set(self.subject_feature_names))
        if missing_subject:
            raise KeyError(
                f"Unknown subject features: {missing_subject}. "
                f"Available subject features: {self.subject_feature_names}"
            )

    def _build_same_dataset_type(
        self,
        participant_datasets: list["SingleParticipantProcessedFeatureDataset"],
    ) -> "FeaturesDataset":
        """
        Reconstruit un dataset du même type que self.dataset.

        Important :
        conserve selected_features si dataset restreint.
        """

        from .base import FeaturesDataset
        from .selected import SelectedFeaturesDataset

        if isinstance(self.dataset, SelectedFeaturesDataset):
            return SelectedFeaturesDataset(
                participant_datasets=participant_datasets,
                selected_features=list(self.dataset.selected_features),
            )

        return FeaturesDataset(participant_datasets)

    # ==========================================================================
    # Sélection de features
    # ==========================================================================

    def select(self, features: Sequence[str]) -> "SelectedFeaturesDataset":

        from .base import FeaturesDataset
        from .participant import SingleParticipantProcessedFeatureDataset
        from .selected import SelectedFeaturesDatasetFactory

        scalar_features, connectivity_features, subject_features = (
            self._split_requested_features(features)
        )

        self._validate_requested_features(
            scalar_features,
            connectivity_features,
            subject_features,
        )

        selected_band_names = {
            self._connectivity_feature_to_band_name(name)
            for name in connectivity_features
        }

        new_participant_datasets = []

        for participant_dataset in self.dataset.participant_datasets:

            if scalar_features:
                new_features_df = participant_dataset.features_df.loc[
                    :, scalar_features
                ].copy()
            else:
                new_features_df = participant_dataset.features_df.iloc[:, 0:0].copy()

            if selected_band_names:
                new_ppc_band_results = {
                    band: value
                    for band, value in participant_dataset.ppc_band_results.items()
                    if band in selected_band_names
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

        base_dataset = FeaturesDataset(new_participant_datasets)

        return SelectedFeaturesDatasetFactory.from_feature_family_names(
            dataset=base_dataset,
            feature_family_names=list(features),
        )

    def select_features(self, features: Sequence[str]) -> "SampleSelector":
        return self._clone_with_dataset(self.select(features))

    def drop(self, features: Sequence[str]) -> "SelectedFeaturesDataset":

        features_to_drop = set(self._ensure_non_empty_list(features, name="features"))

        unknown = sorted(features_to_drop - set(self.selectable_feature_names))
        if unknown:
            raise KeyError(
                f"Unknown features: {unknown}. "
                f"Available selectable features: {self.selectable_feature_names}"
            )

        kept_features = [
            f for f in self.selectable_feature_names if f not in features_to_drop
        ]

        if not kept_features:
            raise ValueError("Dropping these features would leave no feature.")

        return self.select(kept_features)

    def drop_features(self, features: Sequence[str]) -> "SampleSelector":
        return self._clone_with_dataset(self.drop(features))

    # ==========================================================================
    # Sélection participants
    # ==========================================================================

    def select_subject_ids(self, subject_ids):

        subject_ids = self._ensure_non_empty_list(subject_ids, name="subject_ids")
        subject_ids = set(subject_ids)

        new_participants = [
            p
            for p in self.dataset.participant_datasets
            if p.subject.id in subject_ids
        ]

        if not new_participants:
            raise ValueError("No participant matched requested subject_ids.")

        return self._build_same_dataset_type(new_participants)

    def select_subject_ids_selector(self, subject_ids):
        return self._clone_with_dataset(self.select_subject_ids(subject_ids))

    # ==========================================================================
    # Filtrage santé
    # ==========================================================================

    def filter_by_healthstate(self, healthstates):

        parsed = [
            EnumParser.parse(h, HealthState).value
            for h in healthstates
        ]

        new_participants = [
            p
            for p in self.dataset.participant_datasets
            if p.subject.health_state in parsed
        ]

        if not new_participants:
            raise ValueError("No participant matched requested healthstates.")

        return self._build_same_dataset_type(new_participants)

    def filter_by_healthstate_selector(self, healthstates):
        return self._clone_with_dataset(self.filter_by_healthstate(healthstates))

    # ==========================================================================
    # Splits
    # ==========================================================================

    def group_train_test_split(
        self,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    ):

        subject_ids = list(
            dict.fromkeys(
                p.subject.id for p in self.dataset.participant_datasets
            )
        )

        if len(subject_ids) < 2:
            raise ValueError("At least 2 groups required.")

        train_ids, test_ids = train_test_split(
            subject_ids,
            test_size=test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        return (
            self.select_subject_ids(train_ids),
            self.select_subject_ids(test_ids),
        )

    def group_train_val_test_split(
        self,
        val_size=0.2,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    ):

        subject_ids = list(
            dict.fromkeys(
                p.subject.id for p in self.dataset.participant_datasets
            )
        )

        if len(subject_ids) < 3:
            raise ValueError("At least 3 groups required.")

        train_ids, temp_ids = train_test_split(
            subject_ids,
            test_size=val_size + test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        relative_test_size = test_size / (val_size + test_size)

        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=relative_test_size,
            random_state=random_state,
            shuffle=shuffle,
        )

        return (
            self.select_subject_ids(train_ids),
            self.select_subject_ids(val_ids),
            self.select_subject_ids(test_ids),
        )