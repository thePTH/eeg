from __future__ import annotations

import numpy as np
import pandas as pd

from features.factory import CompleteFeatureExtractionResult
from features.results import (
    FeatureExtractionResult,
    PPCBandExtractionResult,
    PSDBandExtractionResult,
)

from .participant import SingleParticipantProcessedFeatureDataset


class SingleParticipantProcessedFeatureDatasetFactory:
    """
    Factory responsible for building a SingleParticipantProcessedFeatureDataset
    from a complete feature extraction result.

    Role
    ----
    This class centralizes the conversion between extraction result objects
    produced by the pipeline and the subject-level dataset format used in the
    rest of the project.

    It guarantees:
    - consistent conversion of scalar features into a DataFrame;
    - consistent conversion of PSD results into JSON-compatible dictionaries;
    - compact conversion of PPC matrices into ``numpy.ndarray[float32]``.
    """

    @staticmethod
    def build(
        complete_extraction_result: CompleteFeatureExtractionResult,
    ) -> SingleParticipantProcessedFeatureDataset:
        """
        Build a SingleParticipantProcessedFeatureDataset from a complete extraction result.

        Parameters
        ----------
        complete_extraction_result
            Object grouping all extraction results required for one participant:
            scalar features, PSD, PPC, subject metadata, and EEG information.

        Returns
        -------
        SingleParticipantProcessedFeatureDataset
            Subject-level dataset ready to be used in the rest of the pipeline.
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

            _eeg=complete_extraction_result.feature_result.eeg
        )

    @staticmethod
    def _build_features_df(
        feature_result: FeatureExtractionResult,
    ) -> pd.DataFrame:
        """
        Convert scalar feature extraction results to a channels-by-features DataFrame.

        Implementation choices
        ----------------------
        - A defensive copy is created.
        - Values are converted to ``float32`` to reduce memory usage.

        Parameters
        ----------
        feature_result
            Scalar feature extraction result.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``float32`` values.
        """
        df = feature_result.dataframe.copy()

        return df.astype(np.float32, copy=False)

    @staticmethod
    def _build_psd_dict(
        psd_result: PSDBandExtractionResult,
    ) -> dict[str, dict[str, float]]:
        """
        Convert PSD results to a dictionary of native Python floats.

        Output format
        -------------
        {
            "Fp1": {"delta": 0.12, "theta": 0.08, ...},
            "Fp2": {"delta": 0.10, "theta": 0.07, ...},
            ...
        }

        Native Python floats simplify potential JSON serialization.

        Parameters
        ----------
        psd_result
            PSD extraction result.

        Returns
        -------
        dict[str, dict[str, float]]
            JSON-friendly PSD dictionary.
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
        Convert PPC results to ``float32`` NumPy matrices.

        Output format
        -------------
        {
            "alpha": ndarray[n_channels, n_channels],
            "beta": ndarray[n_channels, n_channels],
            ...
        }

        Implementation choices
        ----------------------
        Matrices are stored as ``float32`` to:
        - reduce memory usage;
        - speed up some NumPy operations;
        - keep a homogeneous format across the project.

        Parameters
        ----------
        ppc_result
            PPC extraction result.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of PPC matrices.
        """
        return {
            band_name: np.asarray(ppc_result.matrix(band_name), dtype=np.float32)
            for band_name in ppc_result.band_names
        }