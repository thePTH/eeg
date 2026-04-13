from __future__ import annotations

import mne

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class CropStep(PreprocessingStep):
    def __init__(self, tmin: float, tmax: float | None = None):
        self._tmin = tmin
        self._tmax = tmax

    @property
    def name(self) -> str:
        return "crop"

    @property
    def params(self) -> dict:
        return {
            "tmin": self._tmin,
            "tmax": self._tmax,
        }

    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        raw.crop(self._tmin, self._tmax, verbose=False)
        return raw