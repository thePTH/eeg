from __future__ import annotations

import mne

from eeg.data import EEGData
from preprocessing.step.base import PreprocessingStep


class BandpassFilterStep(PreprocessingStep):
    """Band-pass filtering preprocessing step."""

    def __init__(self, band: tuple[float, float]):
        """
        Initialize the band-pass filter.

        Parameters
        ----------
        band
            Tuple containing the lower and upper cutoff frequencies
            in Hertz: ``(l_freq, h_freq)``.
        """
        self._band = band

    @property
    def name(self) -> str:
        """Return the preprocessing step name."""
        return "bandpass_filter"

    @property
    def params(self) -> dict:
        """Return the filter parameters."""
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
        """
        Apply the band-pass filter to an MNE Raw object.

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
            The filtered raw object.
        """
        l_freq, h_freq = self._band

        raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            verbose=False,
        )

        return raw