from __future__ import annotations

import numpy as np
import pandas as pd

from features.factory import CompleteFeatureExtractionResult
from features.results import (
    FeatureExtractionResult,
    PSDBandExtractionResult,
    PPCBandExtractionResult,
)

from .participant import SingleParticipantProcessedFeatureDataset


class SingleParticipantProcessedFeatureDatasetFactory:
    """
    Factory chargée de construire un
    `SingleParticipantProcessedFeatureDataset`
    à partir d'un résultat complet d'extraction de features.

    Rôle
    ----
    Cette classe centralise la conversion entre les objets "résultats
    d'extraction" du pipeline et le format dataset sujet-level utilisé
    dans la suite du projet.

    Elle garantit notamment :
    - une conversion cohérente des features scalaires en `DataFrame`,
    - une conversion cohérente des PSD en dictionnaires JSON-compatibles,
    - une conversion compacte des matrices PPC en `numpy.ndarray[float32]`.
    """

    @staticmethod
    def build(
        complete_extraction_result: CompleteFeatureExtractionResult,
    ) -> SingleParticipantProcessedFeatureDataset:
        """
        Construit un `SingleParticipantProcessedFeatureDataset`
        à partir d'un résultat complet d'extraction.

        Parameters
        ----------
        complete_extraction_result:
            Objet regroupant tous les résultats d'extraction nécessaires
            pour un participant :
            - features scalaires,
            - PSD,
            - PPC,
            - métadonnées sujet,
            - informations EEG.

        Returns
        -------
        SingleParticipantProcessedFeatureDataset
            Dataset sujet-level prêt à être utilisé dans le reste du pipeline.
        """
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
            pipeline_name=str(
                complete_extraction_result.feature_result.eeg.pipeline_name
            ),
            eeg_info_dico=dict(
                complete_extraction_result.feature_result.eeg_info_dico
            ),
        )

    @staticmethod
    def _build_features_df(
        feature_result: FeatureExtractionResult,
    ) -> pd.DataFrame:
        """
        Convertit le résultat d'extraction des features scalaires
        en `DataFrame` de forme [channels x features].

        Choix d'implémentation
        ----------------------
        - on effectue une copie défensive,
        - on convertit en `float32` pour réduire l'empreinte mémoire.

        Parameters
        ----------
        feature_result:
            Résultat d'extraction des features scalaires.

        Returns
        -------
        pd.DataFrame
            DataFrame typé `float32`.
        """
        df = feature_result.dataframe.copy()
        return df.astype(np.float32, copy=False)

    @staticmethod
    def _build_psd_dict(
        psd_result: PSDBandExtractionResult,
    ) -> dict[str, dict[str, float]]:
        """
        Convertit les résultats PSD en dictionnaire de floats Python.

        Format de sortie
        ----------------
        {
            "Fp1": {"delta": 0.12, "theta": 0.08, ...},
            "Fp2": {"delta": 0.10, "theta": 0.07, ...},
            ...
        }

        Remarque
        --------
        On garde ici des `float` Python natifs, ce qui simplifie la
        sérialisation éventuelle en JSON.

        Parameters
        ----------
        psd_result:
            Résultat d'extraction PSD.

        Returns
        -------
        dict[str, dict[str, float]]
            Dictionnaire PSD sérialisable simplement.
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
        Convertit les résultats PPC en matrices numpy `float32`.

        Format de sortie
        ----------------
        {
            "alpha": ndarray[n_channels, n_channels],
            "beta": ndarray[n_channels, n_channels],
            ...
        }

        Choix d'implémentation
        ----------------------
        On stocke les matrices en `float32` pour :
        - réduire l'empreinte mémoire,
        - accélérer certaines opérations numpy,
        - conserver un format homogène dans tout le projet.

        Parameters
        ----------
        ppc_result:
            Résultat d'extraction PPC.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionnaire de matrices PPC.
        """
        return {
            band_name: np.asarray(ppc_result.matrix(band_name), dtype=np.float32)
            for band_name in ppc_result.band_names
        }