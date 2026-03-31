from eeg.data import EEGRecordedData, EEGProcessedData
from preprocessing.step.base import PreprocessingStep

class PreprocessingPipeline:
    def __init__(self, name: str, steps: list[PreprocessingStep]):
        if not steps:
            raise ValueError("A pipeline must contain at least one step")

        self._name = name
        self._steps = steps

    @property
    def name(self):
        return self._name

    @property
    def steps(self):
        return self._steps

    def compute(self, recorded_data: EEGRecordedData) -> EEGProcessedData:
        current_eeg = recorded_data.copy()
        
        for step in self.steps:
            current_eeg = step.transform(current_eeg)
            
        return EEGProcessedData(
            raw=current_eeg.raw,
            source=recorded_data,
            pipeline_name=self.name
        )