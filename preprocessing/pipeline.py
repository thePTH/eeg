from __future__ import annotations

from eeg.data import EEGRecordedData, EEGProcessedData
from preprocessing.step.base import PreprocessingStep


class PreprocessingPipeline:
    """
    Pipeline de preprocessing EEG optimisé.

    Gains principaux
    ----------------
    - une seule copie de travail du Raw
    - plus de copie complète à chaque step
    - possibilité de préparer les steps une seule fois
    - possibilité de réutiliser les caches internes des steps
    """

    def __init__(self, name: str, steps: list[PreprocessingStep]):
        if not steps:
            raise ValueError("A pipeline must contain at least one step")

        self._name = name
        self._steps = steps

    @property
    def name(self) -> str:
        return self._name

    @property
    def steps(self) -> list[PreprocessingStep]:
        return self._steps

    def describe(self) -> dict:
        return {
            "pipeline_name": self.name,
            "steps": [step.describe() for step in self.steps],
        }

    def prepare(self, recorded_data: EEGRecordedData) -> None:
        """
        Prépare toutes les steps sur un EEG source.

        Très utile quand une step lourde, comme ASR, doit construire
        une calibration réutilisable.
        """
        with recorded_data.loaded():
            for step in self.steps:
                step.prepare(recorded_data)

    def clear_caches(self) -> None:
        for step in self.steps:
            step.clear_cache()

    def compute(
        self,
        recorded_data: EEGRecordedData,
        *,
        unload_source: bool = True,
        prepare_steps: bool = True,
    ) -> EEGProcessedData:
        """
        Calcule l'EEG preprocessé à partir d'un EEG brut.

        Stratégie
        ---------
        1. charge la source si nécessaire
        2. prépare éventuellement les steps
        3. crée UNE seule copie de travail du Raw
        4. applique les steps sur cette copie
        5. décharge la source si demandé
        """
        was_loaded_before = recorded_data.is_loaded
        recorded_data.load()

        try:
            if prepare_steps:
                for step in self.steps:
                    step.prepare(recorded_data)

            current_raw = recorded_data.raw.copy()

            for step in self.steps:
                current_raw = step.transform_raw(
                    current_raw,
                    eeg_data=recorded_data,
                )

            return EEGProcessedData(
                raw=current_raw,
                source=recorded_data,
                pipeline_name=self.name,
            )

        finally:
            if unload_source and not was_loaded_before:
                recorded_data.unload()