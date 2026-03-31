from preprocessing.step.base import PreprocessingStep, EEGData


class CropStep(PreprocessingStep):
    def __init__(self, tmin: float, tmax:float=None):
        self._tmin = tmin
        self._tmax = tmax

    @property
    def name(self) -> str:
        return "crop"

    @property
    def params(self) -> dict:
        return {"tmin": self._tmin, "tmax":self._tmax}

    def transform(self, eeg_data):
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw
        new_raw = raw.crop(self._tmin, self._tmax, verbose = False)
        new_eeg._raw = new_raw


        return new_eeg