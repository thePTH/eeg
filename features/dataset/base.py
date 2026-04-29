from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.dataframe import DataframeHelpers

if TYPE_CHECKING:
    from .participant import SingleParticipantProcessedFeatureDataset
    from .selector import SampleSelector


class FeaturesDataset:
    """
    Dataset global regroupant plusieurs participants déjà transformés
    en objets `SingleParticipantProcessedFeatureDataset`.

    Ce dataset centralise les différentes vues utiles du projet :
    - des vues sujet-level (métadonnées),
    - des vues wide adaptées au machine learning,
    - des vues long adaptées à l'analyse,
    - des agrégations moyennes inter-sujets.

    Convention de nommage
    ---------------------
    - Features scalaires wide :
      <channel>_<feature_name>
      Exemple : Fp1_entropy, Cz_theta_beta_ratio

    - Features de connectivité wide :
      cn_<band>_<seed>_<target>
      Exemple : cn_alpha_Fp1_Fp2

    Remarques
    ---------
    - Les colonnes de métadonnées sujet sont présentes dans `wide_dataframe`
      mais ne sont pas toutes utilisées dans `X`.
    - `subject_health` est considéré comme la target.
    - `subject_id` est considéré comme un identifiant / groupe, pas comme
      une variable explicative.
    """

    CONNECTIVITY_PREFIX = "cn_"

    # Certaines bandes de connectivité peuvent exister dans les résultats
    # bruts mais ne doivent pas forcément être exposées au ML.
    EXCLUDED_CONNECTIVITY_BANDS = {"full"}

    SUBJECT_METADATA_COLUMNS = [
        "subject_id",
        "subject_health",
        "subject_group",
        "subject_gender",
        "subject_mmse",
        "subject_age",
    ]

    # Colonnes sujet explicitement autorisées comme variables explicatives.
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
        Paramètres
        ----------
        participant_datasets:
            Liste non vide de datasets sujet-level.
        """
        if not participant_datasets:
            raise ValueError("participant_datasets cannot be empty.")

        self.participant_datasets = participant_datasets

    # ==========================================================================
    # Accès simples aux objets métier
    # ==========================================================================

    @property
    def subjects(self):
        """
        Liste des objets sujet.
        """
        return [dataset.subject for dataset in self.participant_datasets]

    @property
    def ch_names(self) -> list[str]:
        """
        Noms des canaux EEG.

        Hypothèse
        ---------
        Tous les participants partagent le même espace de canaux.
        """
        return self.participant_datasets[0].ch_names

    @property
    def eeg_info(self):
        """
        Informations EEG globales, prises depuis le premier participant.

        Hypothèse
        ---------
        Tous les participants partagent la même structure EEG.
        """
        return self.participant_datasets[0].eeg_info

    @property
    def pipeline_name(self) -> str:
        """
        Nom du pipeline utilisé pour produire les features.
        """
        return self.participant_datasets[0].pipeline_name

    @property
    def groups(self):
        """
        Groupes d'échantillons au niveau wide.
        Ici : `subject_id`.
        """
        return self.wide_dataframe["subject_id"]

    @property
    def selector(self) -> "SampleSelector":
        """
        Point d'entrée vers la logique de sélection / filtrage / split.
        """
        from .selector import SampleSelector

        return SampleSelector(self)

    def participant_dataset(
        self,
        participant_id: str,
    ) -> "SingleParticipantProcessedFeatureDataset":
        """
        Retourne le dataset d'un participant à partir de son identifiant.

        Raises
        ------
        KeyError
            Si aucun participant ne correspond à `participant_id`.
        """
        for dataset in self.participant_datasets:
            if dataset.subject.id == participant_id:
                return dataset

        raise KeyError(f"No participant dataset found for id='{participant_id}'.")

    # ==========================================================================
    # Noms de familles de features (niveau métier)
    # ==========================================================================

    @property
    def scalar_feature_names(self) -> list[str]:
        """
        Familles de features scalaires disponibles au niveau dataset.

        Exemple
        -------
        - variance
        - entropy
        - theta_beta_ratio
        """
        return self.participant_datasets[0].feature_names

    @property
    def psd_band_names(self) -> list[str]:
        """
        Noms des bandes PSD disponibles.
        """
        return self.participant_datasets[0].psd_band_names

    @property
    def ppc_band_names(self) -> list[str]:
        """
        Noms des bandes PPC disponibles.
        """
        return self.participant_datasets[0].ppc_band_names

    @property
    def connectivity_band_names(self) -> list[str]:
        """
        Bandes de connectivité conservées pour les vues ML.

        Certaines bandes peuvent être exclues volontairement
        (exemple : "full").
        """
        return [
            band
            for band in self.ppc_band_names
            if band not in self.EXCLUDED_CONNECTIVITY_BANDS
        ]

    @property
    def connectivity_feature_names(self) -> list[str]:
        """
        Familles de features de connectivité disponibles au niveau dataset.

        Exemple
        -------
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
        Features sujet autorisées comme variables explicatives.

        Exemple
        -------
        - subject_group
        - subject_gender
        - subject_mmse
        - subject_age
        """
        return list(self.SUBJECT_FEATURE_COLUMNS)

    @property
    def feature_names(self) -> list[str]:
        """
        Noms de familles de features disponibles pour la sélection métier.
        """
        return (
            self.scalar_feature_names
            + self.connectivity_feature_names
            + self.subject_feature_names
        )

    @property
    def ppc_edge_keys(self) -> list[str]:
        """
        Clés canoniques des arêtes de connectivité.
        """
        return self.participant_datasets[0].ppc_edge_keys

    # ==========================================================================
    # Helpers internes de construction des vues wide
    # ==========================================================================

    @staticmethod
    def _edge_to_column_suffix(seed: str, target: str) -> str:
        """
        Construit le suffixe d'une colonne wide de connectivité.
        """
        return f"{seed}_{target}"

    @cached_property
    def _scalar_value_columns(self) -> list[str]:
        """
        Colonnes wide correspondant aux features scalaires.

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
        Colonnes wide correspondant aux features de connectivité.

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

    # ==========================================================================
    # Vues tabulaires sujet-level
    # ==========================================================================

    @cached_property
    def subject_dataframe(self) -> pd.DataFrame:
        """
        Vue tabulaire sujet-level des métadonnées participants.
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

        df = pd.DataFrame(rows)

        for column in ["subject_health", "subject_group", "subject_gender"]:
            if column in df.columns:
                df[column] = df[column].astype("category")

        return df

    # ==========================================================================
    # Vues wide
    # ==========================================================================

    @cached_property
    def wide_scalar_dataframe(self) -> pd.DataFrame:
        """
        Vue wide contenant :
        - les métadonnées sujet,
        - toutes les features scalaires vectorisées.

        Chaque ligne correspond à un participant.
        """
        data = np.vstack(
            [
                dataset.features_df.to_numpy(dtype=np.float32, copy=False).ravel(order="C")
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
        Vue wide contenant :
        - les métadonnées sujet,
        - toutes les connectivités vectorisées.

        Chaque ligne correspond à un participant.
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
        Vue wide complète du dataset.

        Contenu
        -------
        - métadonnées sujet,
        - features scalaires,
        - features de connectivité.
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

    # ==========================================================================
    # Vues long
    # ==========================================================================

    @cached_property
    def long_dataframe(self) -> pd.DataFrame:
        """
        Vue longue des features scalaires.

        Colonnes principales
        --------------------
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
        """
        Vue longue des PSD.
        """
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
        """
        Vue longue des connectivités PPC.
        """
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
        """
        Vue longue des connectivités effectivement retenues comme features ML.
        """
        df = self.long_ppc_dataframe
        df = df.loc[~df["band"].isin(self.EXCLUDED_CONNECTIVITY_BANDS)].copy()
        df["connectivity_feature"] = self.CONNECTIVITY_PREFIX + df["band"].astype(str)
        return df

    # ==========================================================================
    # Agrégations
    # ==========================================================================

    @cached_property
    def mean_feature_df(self) -> pd.DataFrame:
        """
        Moyenne inter-sujets des features scalaires.
        """
        return DataframeHelpers.mean(
            [dataset.features_df for dataset in self.participant_datasets]
        )

    @cached_property
    def mean_psd_df(self) -> pd.DataFrame:
        """
        Moyenne inter-sujets des PSD.
        """
        return DataframeHelpers.mean(
            [dataset.to_psd_dataframe() for dataset in self.participant_datasets]
        )

    # ==========================================================================
    # Interface ML
    # ==========================================================================

    @cached_property
    def all_feature_names(self) -> list[str]:
        """
        Toutes les colonnes de la vue wide.
        """
        return list(self.wide_dataframe.columns)

    @cached_property
    def X(self) -> pd.DataFrame:
        """
        Matrice explicative par défaut.

        Colonnes exclues
        ----------------
        - subject_id     : identifiant
        - subject_health : variable cible
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
        """
        Variable cible de classification.
        """
        return self.wide_dataframe["subject_health"]


    def select_rows(self, row_indices) -> "FeaturesDataset":
       
        if row_indices is None:
            raise ValueError("`row_indices` cannot be None.")

        selected_participants = [
            self.participant_datasets[int(i)]
            for i in row_indices
        ]

        if not selected_participants:
            raise ValueError("Row selection produced an empty dataset.")

        return FeaturesDataset(participant_datasets=selected_participants)