from __future__ import annotations

import mne

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class CropStep(PreprocessingStep):
    """
    Preprocessing step that crops an EEG recording to a given time interval.
    """

    def __init__(self, tmin: float, tmax: float | None = None):
        """
        Initialize the cropping step.

        Parameters
        ----------
        tmin
            Start time of the retained interval, in seconds.
        tmax
            End time of the retained interval, in seconds.
            If ``None``, the recording is cropped until its end.
        """
        self._tmin = tmin
        self._tmax = tmax

    @property
    def name(self) -> str:
        """Return the preprocessing step name."""
        return "crop"

    @property
    def params(self) -> dict:
        """Return the cropping parameters."""
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
        """
        Crop the input MNE Raw object.

        Parameters
        ----------
        raw
            Input raw EEG recording.
        eeg_data
            Source EEG object. It is accepted for API consistency but is not
            used by this preprocessing step.

        Returns
        -------
        mne.io.Raw
            The cropped Raw object.
        """
        raw.crop(
            self._tmin,
            self._tmax,
            verbose=False,
        )

        return raw