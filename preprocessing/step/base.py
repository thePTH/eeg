from abc import ABC, abstractmethod
from eeg.data import EEGData

class PreprocessingStep(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def params(self) -> dict:
        return {}

    @abstractmethod
    def transform(self, eeg_data:EEGData)->EEGData:
        pass

    def describe(self) -> dict:
        return {
            "step_name": self.name,
            "params": self.params
        }