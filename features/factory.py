from features.config import FeatureExtractionConfig
from features.context import FeatureExtractionContext
from eeg.signal import SampledSignal, SignalAnalysisEngine
from maths.engines.spectral import PSDAnalysisResult

from eeg.data import EEGProcessedData
import pandas as pd

from features.definitions import complexity, entropy, power_ratios, spectral, temporal, wavelets

from features.categories import FeatureCategory
from features.definitions.base import EEGExtractedFeature, RegisteredFeatureProvider

from eeg.ppc import SignalPPCAnalysisResult, PPCAnalysisEngineParametersFactory, SignalPPCAnalysisEngine




class FeatureExtractionResult:
    def __init__(self, eeg:EEGProcessedData, extraction_config:FeatureExtractionConfig, features_dico:dict[SampledSignal, list[EEGExtractedFeature]], psd_result_dico:dict[SampledSignal, PSDAnalysisResult], ppc_result:SignalPPCAnalysisResult):
        self._eeg = eeg
        self._features_dico = features_dico
        self._config = extraction_config
        self._psd_result_dico = psd_result_dico
        self._ppc_result = ppc_result

    @property
    def eeg(self):
        return self._eeg
    
    @property
    def ppc_result(self):
        return self._ppc_result
    
    @property
    def feature_names(self):
        extracted_features = list(self._features_dico.values())[0]
        return [extracted_feature.name for extracted_feature in extracted_features]
    
    @property
    def dico(self) -> dict[str, dict[str, float]]:
        values_dico = {}
        for signal, extracted_features in self._features_dico.items():
            signal_feature= {extraced_feature.name : extraced_feature.value for extraced_feature in extracted_features}
            values_dico[signal.name] = signal_feature
        return values_dico
    
    @property
    def dataframe(self) :
        return pd.DataFrame.from_dict(self.dico, orient="index")
    
    
    def values(self, feature_name:str):
        dico = self.dataframe.to_dict()
        return list(dico[feature_name].values())
    
    @property
    def config(self):
        return self._config
    
    @property
    def psd_results(self):
        return {signal.name : psd_result for signal, psd_result in self._psd_result_dico.items()}
    


class FeatureExtractionEngine:
    def __init__(self, config:FeatureExtractionConfig):
        self.config = config

    def _build_extraction_context(self, signal:SampledSignal):
        analysis_engine = SignalAnalysisEngine(signal=signal, config=self.config)
        analysis_result = analysis_engine.compute()
        return FeatureExtractionContext(analysis_result)
    
    def _extract_features_from_context(self, context:FeatureExtractionContext) -> list[EEGExtractedFeature]:
        categories_to_extract = context.cfg.categories_to_extract if context.cfg.categories_to_extract else list(FeatureCategory) 
        features_extracted : list[EEGExtractedFeature] = []
        for feature in RegisteredFeatureProvider.get_by_categories(categories=categories_to_extract) :
            extracted_feature = feature.compute(context)
            features_extracted.append(extracted_feature)
                
        return features_extracted

    
    def extract(self, eeg:EEGProcessedData):
        features_dico = {}
        psd_dico = {}

        ppc_params = PPCAnalysisEngineParametersFactory.build_ppc_engine_parameters(self.config)
        ppc_engine =SignalPPCAnalysisEngine(ppc_params)
        ppc_result = ppc_engine.compute(eeg)

        for signal in eeg.signals :
            #print(f"Traitement du signal {signal.name}")
            context = self._build_extraction_context(signal)
            extracted_signal_features_dico = self._extract_features_from_context(context)
            psd_dico[signal] = context.spectral.psd_analysis_result
            features_dico[signal] = extracted_signal_features_dico
            #print("")

        return FeatureExtractionResult(eeg, self.config, features_dico, psd_dico, ppc_result)


