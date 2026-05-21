from __future__ import annotations

from eeg.data import EEGData, EEGProcessedData
from preprocessing.step.base import PreprocessingStep


class PreprocessingPipeline:
    """
    Pipeline de preprocessing EEG optimisé.

    Accepte maintenant :
    - EEGRecordedData
    - EEGProcessedData

    Si on applique un pipeline sur un EEG déjà processed, alors le nouvel
    EEGProcessedData a pour source directe cet EEG processed.

    Exemple :
        recorded
            -> processed_1(source=recorded)
            -> processed_2(source=processed_1)

    On écrase donc bien la source logique par la dernière donnée EEG utilisée.
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

    def prepare(self, eeg_data: EEGData) -> None:
        """
        Prépare toutes les steps sur un EEG source.

        eeg_data peut être :
        - EEGRecordedData
        - EEGProcessedData
        """

        with eeg_data.loaded():
            for step in self.steps:
                step.prepare(eeg_data)

    def clear_caches(self) -> None:
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
        Calcule l'EEG preprocessé à partir de n'importe quel EEGData.

        Stratégie
        ---------
        1. charge l'EEG source si nécessaire
        2. prépare éventuellement les steps
        3. crée une seule copie de travail du Raw
        4. applique les steps sur cette copie
        5. retourne EEGProcessedData(source=eeg_data)

        Important
        ---------
        Si eeg_data est déjà un EEGProcessedData, on ne remonte pas à sa source.
        Le nouveau processed pointe directement vers eeg_data.
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