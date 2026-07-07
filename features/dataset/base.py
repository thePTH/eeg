from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from eeg.data import EEGProcessedData
from utils.dataframe import DataframeHelpers

if TYPE_CHECKING:
    from .participant import SingleParticipantProcessedFeatureDataset
    from .selector import SampleSelector


class FeaturesDataset:
    """
    Global dataset grouping several participants already transformed into
    SingleParticipantProcessedFeatureDataset objects.

    This dataset centralizes the main tabular views used in the project:
    - subject-level metadata views;
    - wide views for machine learning;
    - long views for analysis;
    - inter-subject mean aggregations.

    Naming convention
    -----------------
    Scalar wide features:
        <channel>_<feature_name>
        Example: Fp1_entropy, Cz_theta_beta_ratio

    Connectivity wide features:
        cn_<band>_<seed>_<target>
        Example: cn_alpha_Fp1_Fp2

    Notes
    -----
    - Subject metadata columns are present in ``wide_dataframe`` but are not
      all used in ``X``.
    - ``subject_health`` is considered the target variable.
    - ``subject_id`` is considered an identifier/group, not an explanatory
      variable.
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

    SUBJECT_FEATURE_COLUMNS = [
        "subject_group",
        "subject_gender",
        "subject_mmse",
        "subject_age",
    ]

    def __init__(
        self,
        participant_datasets: list["SingleParticipantProcessedFeatureDataset"],
    ):
        """
        Initialize the dataset.

        Parameters
        ----------
        participant_datasets
            Non-empty list of subject-level datasets.
        """
        if not participant_datasets:
            raise ValueError("participant_datasets cannot be empty.")

        self.participant_datasets = participant_datasets

    @property
    def subjects(self):
        """Return the subject objects."""
        return [dataset.subject for dataset in self.participant_datasets]

    @property
    def ch_names(self) -> list[str]:
        """
        Return EEG channel names.

        Assumption
        ----------
        All participants share the same channel space.
        """
        return self.participant_datasets[0].ch_names

    @property
    def eeg_info(self):
        """
        Return global EEG information from the first participant.

        Assumption
        ----------
        All participants share the same EEG structure.
        """
        return self.participant_datasets[0].eeg_info

    @property
    def pipeline_name(self) -> str:
        """Return the pipeline name used to produce the features."""
        return self.participant_datasets[0].pipeline_name

    @property
    def groups(self):
        """Return wide-level sample groups, here ``subject_id``."""
        return self.wide_dataframe["subject_id"]

    @property
    def selector(self) -> "SampleSelector":
        """Return the entry point for selection, filtering, and splitting logic."""
        from .selector import SampleSelector

        return SampleSelector(self)

    def participant_dataset(
        self,
        participant_id: str,
    ) -> "SingleParticipantProcessedFeatureDataset":
        """
        Return the dataset associated with a participant identifier.

        Raises
        ------
        KeyError
            If no participant matches ``participant_id``.
        """
        for dataset in self.participant_datasets:
            if dataset.subject.id == participant_id:
                return dataset

        raise KeyError(f"No participant dataset found for id='{participant_id}'.")

    @property
    def scalar_feature_names(self) -> list[str]:
        """
        Return available scalar feature families.

        Examples
        --------
        - variance
        - entropy
        - theta_beta_ratio
        """
        return self.participant_datasets[0].feature_names

    @property
    def psd_band_names(self) -> list[str]:
        """Return available PSD band names."""
        return self.participant_datasets[0].psd_band_names

    @property
    def ppc_band_names(self) -> list[str]:
        """Return available PPC band names."""
        return self.participant_datasets[0].ppc_band_names

    @property
    def connectivity_band_names(self) -> list[str]:
        """
        Return connectivity bands kept for machine-learning views.

        Some bands may be deliberately excluded, for example ``"full"``.
        """
        return [
            band
            for band in self.ppc_band_names
            if band not in self.EXCLUDED_CONNECTIVITY_BANDS
        ]

    @property
    def connectivity_feature_names(self) -> list[str]:
        """
        Return available connectivity feature families.

        Examples
        --------
        - cn_delta
        - cn_theta
        - cn_alpha
        """
        return [
            f"{self.CONNECTIVITY_PREFIX}{band}"
            for band in self.connectivity_band_names
        ]

    @property
    def subject_feature_names(self) -> list[str]:
        """
        Return subject-level features allowed as explanatory variables.

        Examples
        --------
        - subject_group
        - subject_gender
        - subject_mmse
        - subject_age
        """
        return list(self.SUBJECT_FEATURE_COLUMNS)

    @property
    def feature_names(self) -> list[str]:
        """Return feature family names available for domain-level selection."""
        return (
            self.scalar_feature_names
            + self.connectivity_feature_names
            + self.subject_feature_names
        )

    @property
    def ppc_edge_keys(self) -> list[str]:
        """Return canonical connectivity edge keys."""
        return self.participant_datasets[0].ppc_edge_keys

    @staticmethod
    def _edge_to_column_suffix(seed: str, target: str) -> str:
        """Build the suffix of a wide connectivity column."""
        return f"{seed}_{target}"

    @cached_property
    def _scalar_value_columns(self) -> list[str]:
        """
        Return wide columns corresponding to scalar features.

        Format
        ------
        <channel>_<feature_name>
        """
        first = self.participant_datasets[0].features_df

        return [
            f"{channel}_{feature}"
            for channel in first.index
            for feature in first.columns
        ]

    @cached_property
    def _connectivity_value_columns(self) -> list[str]:
        """
        Return wide columns corresponding to connectivity features.

        Format
        ------
        cn_<band>_<seed>_<target>
        """
        first = self.participant_datasets[0]
        ii, jj = first.ppc_upper_triangle_indices
        ch_names = first.ch_names

        columns: list[str] = []

        for band in self.connectivity_band_names:
            for i, j in zip(ii.tolist(), jj.tolist()):
                columns.append(
                    f"{self.CONNECTIVITY_PREFIX}{band}_{ch_names[i]}_{ch_names[j]}"
                )

        return columns

    @cached_property
    def subject_dataframe(self) -> pd.DataFrame:
        """Return a subject-level tabular view of participant metadata."""
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

        df = pd.DataFrame(rows)

        for column in ["subject_health", "subject_group", "subject_gender"]:
            if column in df.columns:
                df[column] = df[column].astype("category")

        return df

    @cached_property
    def wide_scalar_dataframe(self) -> pd.DataFrame:
        """
        Return a wide view containing subject metadata and scalar features.

        Each row corresponds to one participant.
        """
        data = np.vstack(
            [
                dataset.features_df.to_numpy(dtype=np.float32, copy=False).ravel(
                    order="C"
                )
                for dataset in self.participant_datasets
            ]
        )

        values_df = pd.DataFrame(data, columns=self._scalar_value_columns)

        return pd.concat(
            [
                self.subject_dataframe.reset_index(drop=True),
                values_df.reset_index(drop=True),
            ],
            axis=1,
        )

    @cached_property
    def wide_connectivity_dataframe(self) -> pd.DataFrame:
        """
        Return a wide view containing subject metadata and vectorized connectivity.

        Each row corresponds to one participant.
        """
        first = self.participant_datasets[0]
        ii, jj = first.ppc_upper_triangle_indices

        rows = []

        for dataset in self.participant_datasets:
            row_arrays = []

            for band in self.connectivity_band_names:
                matrix = dataset.ppc_matrix(band, dtype=np.float32)
                row_arrays.append(matrix[ii, jj])

            if row_arrays:
                rows.append(np.concatenate(row_arrays, axis=0))
            else:
                rows.append(np.empty((0,), dtype=np.float32))

        data = np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32)
        values_df = pd.DataFrame(data, columns=self._connectivity_value_columns)

        return pd.concat(
            [
                self.subject_dataframe.reset_index(drop=True),
                values_df.reset_index(drop=True),
            ],
            axis=1,
        )

    @cached_property
    def wide_dataframe(self) -> pd.DataFrame:
        """
        Return the complete wide dataset.

        Content
        -------
        - subject metadata;
        - scalar features;
        - connectivity features.
        """
        scalar_df = self.wide_scalar_dataframe
        connectivity_df = self.wide_connectivity_dataframe.drop(
            columns=self.SUBJECT_METADATA_COLUMNS,
            errors="ignore",
        )

        return pd.concat(
            [
                scalar_df.reset_index(drop=True),
                connectivity_df.reset_index(drop=True),
            ],
            axis=1,
        )

    @cached_property
    def long_dataframe(self) -> pd.DataFrame:
        """
        Return the long view of scalar features.

        Main columns
        ------------
        - channel
        - feature
        - value
        - subject_id
        - subject_health
        - ...
        """
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            features_df = participant_dataset.features_df

            df_long = (
                features_df
                .reset_index(names="channel")
                .melt(
                    id_vars="channel",
                    var_name="feature",
                    value_name="value",
                )
            )

            df_long["subject_id"] = subject.id
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
        """Return the long view of PSD band powers."""
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            psd_df = participant_dataset.to_psd_dataframe()

            df_long = (
                psd_df
                .reset_index(names="channel")
                .melt(
                    id_vars="channel",
                    var_name="band",
                    value_name="value",
                )
            )

            df_long["subject_id"] = subject.id
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
        """Return the long view of PPC connectivity values."""
        rows: list[pd.DataFrame] = []

        for participant_dataset in self.participant_datasets:
            subject = participant_dataset.subject
            df_long = participant_dataset.ppc_edge_dataframe.copy()

            df_long["subject_id"] = subject.id
            df_long["subject_age"] = subject.age
            df_long["subject_health"] = subject.health_state
            df_long["subject_group"] = subject.group
            df_long["subject_gender"] = subject.gender
            df_long["subject_mmse"] = subject.mmse

            rows.append(df_long)

        return pd.concat(rows, ignore_index=True)

    @cached_property
    def long_connectivity_dataframe(self) -> pd.DataFrame:
        """Return the long view of connectivity values kept as ML features."""
        df = self.long_ppc_dataframe
        df = df.loc[~df["band"].isin(self.EXCLUDED_CONNECTIVITY_BANDS)].copy()
        df["connectivity_feature"] = self.CONNECTIVITY_PREFIX + df["band"].astype(str)

        return df

    @cached_property
    def mean_feature_df(self) -> pd.DataFrame:
        """Return the inter-subject mean of scalar features."""
        return DataframeHelpers.mean(
            [dataset.features_df for dataset in self.participant_datasets]
        )

    @cached_property
    def mean_psd_df(self) -> pd.DataFrame:
        """Return the inter-subject mean of PSD band powers."""
        return DataframeHelpers.mean(
            [dataset.to_psd_dataframe() for dataset in self.participant_datasets]
        )

    @cached_property
    def all_feature_names(self) -> list[str]:
        """Return all columns of the wide view."""
        return list(self.wide_dataframe.columns)

    @cached_property
    def X(self) -> pd.DataFrame:
        """
        Return the default explanatory matrix.

        Excluded columns
        ----------------
        - subject_id: identifier
        - subject_health: target variable
        """
        excluded_columns = {"subject_id", "subject_health"}

        feature_columns = [
            column
            for column in self.wide_dataframe.columns
            if column not in excluded_columns
        ]

        return self.wide_dataframe[feature_columns]

    @cached_property
    def y(self):
        """Return the classification target variable."""
        return self.wide_dataframe["subject_health"]

    def select_rows(self, row_indices) -> "FeaturesDataset":
        """Return a new dataset containing only the selected row indices."""
        if row_indices is None:
            raise ValueError("`row_indices` cannot be None.")

        selected_participants = [
            self.participant_datasets[int(i)]
            for i in row_indices
        ]

        if not selected_participants:
            raise ValueError("Row selection produced an empty dataset.")

        return FeaturesDataset(participant_datasets=selected_participants)

    @cached_property
    def eegs(self) -> list[EEGProcessedData]:
        """
        Return the EEG objects associated with dataset rows.

        The EEG objects remain lazy: raw data is not loaded here.
        """
        return [dataset.eeg for dataset in self.participant_datasets]

    @cached_property
    def eeg_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame mapping each subject_id to its EEG object."""
        return pd.DataFrame(
            {
                "subject_id": self.subject_dataframe["subject_id"].values,
                "eeg": self.eegs,
            }
        )