from __future__ import annotations

from dataclasses import dataclass

from eeg.data import EEGProcessedData
from eeg.ppc import (
    PPCAnalysisEngineParametersFactory,
    SignalPPCAnalysisEngine,
)
from eeg.signal import SampledSignal, SignalAnalysisEngine
from features.categories import FeatureCategory
from features.config import FeatureExtractionConfig
from features.context import FeatureExtractionContext
from features.definitions import complexity, entropy, power_ratios, spectral, temporal, wavelets
from features.definitions.base import EEGExtractedFeature, RegisteredFeatureProvider
from features.results import (
    FeatureExtractionResult,
    PSDBandExtractionResult,
    PPCBandExtractionResult,
)


@dataclass(frozen=True)
class CompleteFeatureExtractionResult:
    """
    Bundle utilitaire quand on veut calculer les trois blocs d'un coup,
    tout en gardant une séparation stricte des résultats.
    """

    feature_result: FeatureExtractionResult
    psd_result: PSDBandExtractionResult
    ppc_result: PPCBandExtractionResult


class FeatureExtractionEngine:
    """
    Engine dédié aux features scalaires par canal.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def _build_extraction_context(self, signal: SampledSignal) -> FeatureExtractionContext:
        analysis_engine = SignalAnalysisEngine(signal=signal, config=self.config)
        analysis_result = analysis_engine.compute()
        return FeatureExtractionContext(analysis_result)

    def _extract_features_from_context(
        self,
        context: FeatureExtractionContext,
    ) -> list[EEGExtractedFeature]:
        categories_to_extract = (
            context.cfg.categories_to_extract
            if context.cfg.categories_to_extract
            else list(FeatureCategory)
        )

        features_extracted: list[EEGExtractedFeature] = []
        for feature in RegisteredFeatureProvider.get_by_categories(categories=categories_to_extract):
            extracted_feature = feature.compute(context)
            features_extracted.append(extracted_feature)

        return features_extracted

    def extract(self, eeg: EEGProcessedData) -> FeatureExtractionResult:
        features_dico: dict[SampledSignal, list[EEGExtractedFeature]] = {}

        for signal in eeg.signals:
            context = self._build_extraction_context(signal)
            features_dico[signal] = self._extract_features_from_context(context)

        return FeatureExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            features_dico=features_dico,
        )


class PSDBandExtractionEngine:
    """
    Engine dédié au calcul PSD agrégé par bande et par canal.

    Aucun spectre fréquence-par-fréquence n'est exposé dans le résultat final.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract(self, eeg: EEGProcessedData) -> PSDBandExtractionResult:
        band_powers_by_signal: dict[str, dict[str, float]] = {}

        for signal in eeg.signals:
            analysis_result = SignalAnalysisEngine(signal=signal, config=self.config).compute()
            band_powers_by_signal[signal.name] = {
                band_name: float(power)
                for band_name, power in analysis_result.spectral.band_powers.items()
            }

        return PSDBandExtractionResult(
            eeg=eeg,
            extraction_config=self.config,
            band_powers_by_signal=band_powers_by_signal,
        )


class PPCBandExtractionEngine:
    """
    Engine dédié au calcul PPC agrégé par bande.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config

    def extract(self, eeg: EEGProcessedData) -> PPCBandExtractionResult:
        params = PPCAnalysisEngineParametersFactory.build_ppc_engine_parameters(self.config)
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
        )


class CompleteFeatureExtractionEngine:
    """
    Orchestrateur optionnel qui calcule séparément features, PSD et PPC,
    puis renvoie un bundle.
    """

    def __init__(self, config: FeatureExtractionConfig):
        self.config = config
        self.feature_engine = FeatureExtractionEngine(config)
        self.psd_engine = PSDBandExtractionEngine(config)
        self.ppc_engine = PPCBandExtractionEngine(config)

    def extract(self, eeg: EEGProcessedData) -> CompleteFeatureExtractionResult:
        return CompleteFeatureExtractionResult(
            feature_result=self.feature_engine.extract(eeg),
            psd_result=self.psd_engine.extract(eeg),
            ppc_result=self.ppc_engine.extract(eeg),
        )
