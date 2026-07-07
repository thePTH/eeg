from __future__ import annotations

from dataclasses import dataclass

from eeg.data import EEGProcessedData
from eeg.ppc import (
    PPCAnalysisEngineParametersFactory,
    SignalPPCAnalysisEngine,
)
from eeg.signal import SignalAnalysisEngine, SignalAnalysisResults, SampledSignal
from features.categories import FeatureCategory
from features.config import FeatureExtractionConfig
from features.context import FeatureExtractionContext
from features.definitions import complexity, entropy, power_ratios, spectral, temporal, wavelets
from features.definitions.base import EEGExtractedFeature, RegisteredFeatureProvider
from features.results import (
    FeatureExtractionResult,
    PPCBandExtractionResult,
    PSDBandExtractionResult,
)


@dataclass(frozen=True)
class CompleteFeatureExtractionResult:
    """Bundle returned when all feature extraction blocks are computed together."""

    feature_result: FeatureExtractionResult
    psd_result: PSDBandExtractionResult
    ppc_result: PPCBandExtractionResult


class FeatureExtractionEngine:
    """Engine dedicated to scalar channel-wise EEG feature extraction."""

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self._registered_features = self._resolve_registered_features()

    def _resolve_registered_features(self):
        """Return the registered feature classes matching the configured categories."""
        categories_to_extract = (
            self.config.categories_to_extract
            if self.config.categories_to_extract
            else list(FeatureCategory)
        )

        return RegisteredFeatureProvider.get_by_categories(
            categories=categories_to_extract
        )

    def _build_analysis_result(self, signal: SampledSignal) -> SignalAnalysisResults:
        """Build the complete analysis result for one sampled signal."""
        analysis_engine = SignalAnalysisEngine(signal=signal, config=self.config)

        return analysis_engine.compute()

    def _build_extraction_context(self, signal: SampledSignal) -> FeatureExtractionContext:
        """Build the feature extraction context for one sampled signal."""
        analysis_result = self._build_analysis_result(signal)

        return FeatureExtractionContext(analysis_result)

    def _extract_features_from_context(
        self,
        context: FeatureExtractionContext,
    ) -> list[EEGExtractedFeature]:
        """Extract all configured scalar features from a feature extraction context."""
        features_extracted: list[EEGExtractedFeature] = []

        for feature in self._registered_features:
            extracted_feature = feature.compute(context)
            features_extracted.append(extracted_feature)

        return features_extracted

    def extract(self, eeg: EEGProcessedData) -> FeatureExtractionResult:
        """Extract scalar channel-wise EEG features from processed EEG data."""
        eeg_info_dico = eeg.info.to_json_dict()

        features_dico: dict[str, list[EEGExtractedFeature]] = {}

        for signal in eeg.iter_signals():
            context = self._build_extraction_context(signal)
            features_dico[signal.name] = self._extract_features_from_context(context)

        return FeatureExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            features_dico=features_dico,
            eeg_info_dico=eeg_info_dico,
        )


class PSDBandExtractionEngine:
    """Engine dedicated to channel-wise PSD band-power extraction."""

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def _build_analysis_result(self, signal: SampledSignal) -> SignalAnalysisResults:
        """Build the complete analysis result for one sampled signal."""
        analysis_engine = SignalAnalysisEngine(signal=signal, config=self.config)

        return analysis_engine.compute()

    def extract(self, eeg: EEGProcessedData) -> PSDBandExtractionResult:
        """Extract PSD band powers from processed EEG data."""
        eeg_info_dico = eeg.info.to_json_dict()

        band_powers_by_signal: dict[str, dict[str, float]] = {}

        for signal in eeg.iter_signals():
            analysis_result = self._build_analysis_result(signal)
            band_powers_by_signal[signal.name] = {
                band_name: float(power)
                for band_name, power in analysis_result.spectral.band_powers.items()
            }

        return PSDBandExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            band_powers_by_signal=band_powers_by_signal,
            eeg_info_dico=eeg_info_dico,
        )


class PPCBandExtractionEngine:
    """Engine dedicated to PPC band-wise connectivity extraction."""

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract(self, eeg: EEGProcessedData) -> PPCBandExtractionResult:
        """Extract PPC connectivity matrices from processed EEG data."""
        eeg_info_dico = eeg.info.to_json_dict()

        params = PPCAnalysisEngineParametersFactory.build_ppc_engine_parameters(
            self.config
        )
        engine = SignalPPCAnalysisEngine(params)
        signal_ppc_result = engine.compute(eeg)

        matrices_by_band = {
            band_name: signal_ppc_result.band_matrix(band_name)
            for band_name in signal_ppc_result.band_names
        }

        return PPCBandExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            matrices_by_band=matrices_by_band,
            eeg_info_dico=eeg_info_dico,
        )


class CompleteFeatureExtractionEngine:
    """
    Orchestrator computing three extraction blocks separately:
    - scalar EEG features;
    - PSD band powers;
    - PPC band connectivity.

    Important optimization
    ----------------------
    When scalar features and PSD band powers are extracted together, signal
    analysis is computed only once per channel.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.feature_engine = FeatureExtractionEngine(config)
        self.psd_engine = PSDBandExtractionEngine(config)
        self.ppc_engine = PPCBandExtractionEngine(config)

    def extract(self, eeg: EEGProcessedData) -> CompleteFeatureExtractionResult:
        """Extract scalar features, PSD band powers, and PPC matrices."""
        eeg_info_dico = eeg.info.to_json_dict()

        features_dico: dict[str, list[EEGExtractedFeature]] = {}
        band_powers_by_signal: dict[str, dict[str, float]] = {}

        for signal in eeg.iter_signals():
            analysis_result = SignalAnalysisEngine(
                signal=signal,
                config=self.config,
            ).compute()

            context = FeatureExtractionContext(analysis_result)

            features_dico[signal.name] = self.feature_engine._extract_features_from_context(
                context
            )

            band_powers_by_signal[signal.name] = {
                band_name: float(power)
                for band_name, power in analysis_result.spectral.band_powers.items()
            }

        feature_result = FeatureExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            features_dico=features_dico,
            eeg_info_dico=eeg_info_dico,
        )

        psd_result = PSDBandExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            band_powers_by_signal=band_powers_by_signal,
            eeg_info_dico=eeg_info_dico,
        )

        ppc_result = self.ppc_engine.extract(eeg)

        return CompleteFeatureExtractionResult(
            feature_result=feature_result,
            psd_result=psd_result,
            ppc_result=ppc_result,
        )