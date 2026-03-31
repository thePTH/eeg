from preprocessing.step.base import PreprocessingStep

class BandpassFilterStep(PreprocessingStep):
    def __init__(self, band: tuple[float, float]):
        self._band = band

    @property
    def name(self) -> str:
        return "bandpass_filter"

    @property
    def params(self) -> dict:
        return {
            "l_freq": self._band[0],
            "h_freq": self._band[1]
        }

    def transform(self, eeg_data):
        l_freq, h_freq = self._band
        new_eeg = eeg_data.copy()
        raw = new_eeg.raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)

        new_eeg._raw = raw
        return new_eeg