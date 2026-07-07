from __future__ import annotations

from eeg.data import EEGData, EEGProcessedData
from preprocessing.step.base import PreprocessingStep


class PreprocessingPipeline:
    """
    Optimized EEG preprocessing pipeline.

    Accepted inputs:
    - EEGRecordedData
    - EEGProcessedData

    When the pipeline is applied to an already processed EEG object, the newly
    created EEGProcessedData uses that processed EEG object as its direct source.

    Example
    -------
        recorded
            -> processed_1(source=recorded)
            -> processed_2(source=processed_1)

    The logical source is therefore the last EEG object used as pipeline input.
    """

    def __init__(self, name: str, steps: list[PreprocessingStep]):
        if not steps:
            raise ValueError("A pipeline must contain at least one step")

        self._name = name
        self._steps = steps

    @property
    def name(self) -> str:
        """Return the preprocessing pipeline name."""
        return self._name

    @property
    def steps(self) -> list[PreprocessingStep]:
        """Return the preprocessing steps."""
        return self._steps

    def describe(self) -> dict:
        """Return a serializable description of the preprocessing pipeline."""
        return {
            "pipeline_name": self.name,
            "steps": [step.describe() for step in self.steps],
        }

    def prepare(self, eeg_data: EEGData) -> None:
        """
        Prepare all preprocessing steps on a source EEG object.

        ``eeg_data`` can be:
        - EEGRecordedData
        - EEGProcessedData
        """
        with eeg_data.loaded():
            for step in self.steps:
                step.prepare(eeg_data)

    def clear_caches(self) -> None:
        """Clear cached state from all preprocessing steps."""
        for step in self.steps:
            step.clear_cache()

    def compute(
        self,
        eeg_data: EEGData,
        *,
        unload_source: bool = True,
        prepare_steps: bool = True,
    ) -> EEGProcessedData:
        """
        Compute a processed EEG object from any EEGData object.

        Strategy
        --------
        1. Load the source EEG if needed.
        2. Optionally prepare all preprocessing steps.
        3. Create a single working copy of the Raw object.
        4. Apply each step to the working copy.
        5. Return ``EEGProcessedData(source=eeg_data)``.

        Important
        ---------
        If ``eeg_data`` is already an EEGProcessedData, the method does not go
        back to its source. The newly processed object points directly to
        ``eeg_data``.
        """
        was_loaded_before = eeg_data.is_loaded
        eeg_data.load()

        try:
            if prepare_steps:
                for step in self.steps:
                    step.prepare(eeg_data)

            current_raw = eeg_data.raw.copy()

            for step in self.steps:
                current_raw = step.transform_raw(
                    current_raw,
                    eeg_data=eeg_data,
                )

            return EEGProcessedData(
                raw=current_raw,
                source=eeg_data,
                pipeline_name=self.name,
            )

        finally:
            if unload_source and not was_loaded_before:
                eeg_data.unload()