from __future__ import annotations

from abc import ABC, abstractmethod

import mne

from eeg.data import EEGData


class PreprocessingStep(ABC):
    """
    Classe racine des étapes de preprocessing.

    Contrat
    -------
    - `prepare(...)` permet à une étape de se calibrer ou de mettre en cache
      des informations avant le vrai calcul.
    - `transform_raw(...)` transforme un `mne.io.Raw` de travail.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def params(self) -> dict:
        return {}

    def prepare(self, eeg_data: EEGData) -> None:
        """
        Hook optionnel exécuté avant le preprocessing effectif.

        Cas typique :
        - calibration ASR à faire une seule fois
        - construction d'un cache lié au sujet
        """
        return None

    def clear_cache(self) -> None:
        """
        Hook optionnel pour vider un cache interne.
        """
        return None

    @abstractmethod
    def transform_raw(
        self,
        raw: mne.io.Raw,
        *,
        eeg_data: EEGData | None = None,
    ) -> mne.io.Raw:
        pass

    def transform(self, eeg_data: EEGData) -> EEGData:
        """
        API objet de compatibilité.
        """
        self.prepare(eeg_data)

        with eeg_data.loaded() as raw:
            working_raw = raw.copy()
            transformed_raw = self.transform_raw(working_raw, eeg_data=eeg_data)

        return eeg_data.update_raw(transformed_raw)

    def describe(self) -> dict:
        return {
            "step_name": self.name,
            "params": self.params,
        }