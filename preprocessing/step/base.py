from __future__ import annotations

from abc import ABC, abstractmethod

import mne

from eeg.data import EEGData


class PreprocessingStep(ABC):
    """
    Base class for EEG preprocessing steps.

    Contract
    --------
    - ``prepare(...)`` allows a step to calibrate itself or build caches before
      the actual preprocessing.
    - ``transform_raw(...)`` applies the preprocessing operation to a working
      ``mne.io.Raw`` object.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the preprocessing step name."""
        raise NotImplementedError

    @property
    def params(self) -> dict:
        """Return the preprocessing step parameters."""
        return {}

    def prepare(self, eeg_data: EEGData) -> None:
        """
        Optional hook executed before preprocessing.

        Typical use cases
        -----------------
        - calibrating an ASR model once;
        - building a subject-specific cache.
        """
        return None

    def clear_cache(self) -> None:
        """
        Optional hook used to clear any internal cache.
        """
        return None

    @abstractmethod
    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        """
        Transform a working MNE Raw object.

        Parameters
        ----------
        raw
            Working copy of the EEG recording.
        eeg_data
            Source EEG object. This is optional because not every preprocessing
            step requires access to the original dataset.

        Returns
        -------
        mne.io.Raw
            The transformed Raw object.
        """
        raise NotImplementedError

    def transform(self, eeg_data: EEGData) -> EEGData:
        """
        Compatibility object-oriented API.

        This method:
        1. prepares the preprocessing step;
        2. loads the EEG if necessary;
        3. applies the transformation to a copy of the raw recording;
        4. returns a new EEGData object containing the transformed signal.
        """
        self.prepare(eeg_data)

        with eeg_data.loaded() as raw:
            working_raw = raw.copy()
            transformed_raw = self.transform_raw(
                working_raw,
                eeg_data=eeg_data,
            )

        return eeg_data.update_raw(transformed_raw)

    def describe(self) -> dict:
        """
        Return a serializable description of the preprocessing step.
        """
        return {
            "step_name": self.name,
            "params": self.params,
        }