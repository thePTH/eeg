from eeg.signal import SignalAnalysisResults
import numpy as np
class FeatureExtractionContext:
    def __init__(self, signal_analysis_result:SignalAnalysisResults):
        self.analyis_result = signal_analysis_result

    @property
    def x(self):
        return np.array(self.analyis_result.signal.points)
    
    @property
    def cfg(self):
        return self.analyis_result.config

    @property
    def stats(self):
        return self.analyis_result.stats
    
    @property
    def spectral(self):
        return self.analyis_result.spectral
    
    @property
    def wavelet(self):
        return self.analyis_result.wavelet