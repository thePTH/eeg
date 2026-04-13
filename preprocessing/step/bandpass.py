from __future__ import annotations

import mne

from eeg.data import EEGData
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
            "h_freq": self._band[1],
        }

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        l_freq, h_freq = self._band
        raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
        return raw