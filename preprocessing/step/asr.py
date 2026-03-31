from preprocessing.step.base import PreprocessingStep, EEGData
from asrpy import ASR


class ASRStep(PreprocessingStep):
    def __init__(self, cutoff: float):
        self._cutoff = cutoff

    @property
    def name(self) -> str:
        return "asr"

    @property
    def params(self) -> dict:
        return {"cutoff": self._cutoff}

    def transform(self, eeg_data):
        asr = ASR(sfreq=eeg_data.sampling_frequency, cutoff=self._cutoff)
        raw_original = eeg_data.raw.copy()
        asr.fit(raw_original)
        raw_clean = asr.transform(raw_original)
        return EEGData(raw_clean, sampling_frequency=eeg_data.sampling_frequency)