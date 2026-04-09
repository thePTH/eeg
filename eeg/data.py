from abc import ABC, abstractmethod
import mne
from eeg.signal import SampledSignal
from typing import Self
from preprocessing.names import PipelineName
from utils.enum import EnumParser

from participants.definition import Participant

class EEGData(ABC):
    def __init__(self, raw:mne.io.Raw, sampling_frequency:float):
        self._raw = raw
        self._sampling_frequency= sampling_frequency

    @property
    def raw(self):
        return self._raw
    
    @property
    def data(self):
        return self.raw.get_data()
    
    @property
    def sampling_frequency(self):
        return self._sampling_frequency
    
    @property
    def signal_names(self) ->list[str]:
        return self.raw.ch_names
    
    @property
    def signals(self) -> list[SampledSignal] :
        return [SampledSignal(self.sampling_frequency, list(self.data[k]), self.signal_names[k]) for k in range(len(self.data))]
    
    def _copy_kwargs(self) -> dict:
        return {
            "raw": self.raw.copy(),
            "sampling_frequency": self.sampling_frequency,
        }

    def copy(self) -> Self:
        return type(self)(**self._copy_kwargs())
    
    
    
    def plot(self):
        self.raw.plot(verbose=False)

    @property
    def info(self):
        return self.raw.info
    




    
class EEGRecordedData(EEGData):
    def __init__(self, raw:mne.io.Raw, sampling_frequency:float, subject:Participant):
        super().__init__(raw, sampling_frequency)
        self._subject = subject

    @property
    def subject(self):
        return self._subject
    

    def _copy_kwargs(self) -> dict:
        kwargs = super()._copy_kwargs()
        kwargs["subject"] = self.subject
        return kwargs

    def copy(self) -> Self:
        return type(self)(**self._copy_kwargs())
    

class EEGProcessedData(EEGData):
    def __init__(self, raw: mne.io.Raw, source:EEGRecordedData, pipeline_name:PipelineName):
        super().__init__(raw, source.sampling_frequency)
        self._pipeline_name = EnumParser.parse(pipeline_name, PipelineName)
        self._source = source

    @property
    def pipeline_name(self):
        return self._pipeline_name.value
    
    @property
    def source(self):
        return self._source
    

    
    def _copy_kwargs(self) -> dict:
        kwargs = super()._copy_kwargs()
        kwargs["subject"] = self.subject
        return kwargs



    
import csv
from participants.definition import Participant
from utils.enum import EnumParser
from pathlib import Path

from mne_bids import BIDSPath, read_raw_bids

class EEGRecordedDataProvider:

    @staticmethod
    def _extract_subject(row:dict) -> Participant:
        id = row["participant_id"].split("-")[1]
        gender = row["Gender"]
        age = int(row["Age"])
        group =  row["Group"]
        mmse = int(row["MMSE"])

        return Participant(id=id, gender=gender, age=age, group=group, mmse=mmse)
    
    @staticmethod
    def _extract_recorded_eeg(subject:Participant, root:Path, load_data=True):
        bids_path = BIDSPath(
            subject=subject.id,
            task="eyesclosed",
            datatype="eeg",
            root=root
        )
        raw : mne.io.Raw = read_raw_bids(bids_path=bids_path, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, verbose=False)
        if load_data :
            raw.load_data(verbose=False)

        return EEGRecordedData(raw=raw, sampling_frequency=raw.info['sfreq'], subject=subject)




    @staticmethod
    def build(data_file_path:str, load_data=True) -> list[EEGRecordedData]:
        recordings = []
        root = Path(data_file_path)
        

        with open(root/"participants.tsv", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                subject = EEGRecordedDataProvider._extract_subject(row)
                recorded_eeg = EEGRecordedDataProvider._extract_recorded_eeg(subject, root, load_data)
                recordings.append(recorded_eeg)
                

        return recordings
    




    



    





from collections import defaultdict
from random import Random
from typing import Literal


class EEGRecordedDataHelper:
    @staticmethod
    def update_raw(eeg:EEGData, new_raw: mne.io.Raw) -> Self:
        """
        Reconstruit un objet du même type que self, mais avec un nouvel objet Raw.
        """
        kwargs = eeg._copy_kwargs()
        kwargs["raw"] = new_raw
        return type(eeg)(**kwargs)
    


    @staticmethod
    def split(eeg:EEGRecordedData, t_start:int=10,  window_seconds:int=60) -> list[EEGRecordedData] :
        total_duration = eeg.raw.times[-1]

        if t_start >= total_duration:
            return []

        usable_duration = total_duration - t_start
        n_full_windows = int(usable_duration // window_seconds)
        segments:list[EEGRecordedData] =  []

        for i in range(n_full_windows):
            tmin = t_start + i * window_seconds
            tmax = tmin + window_seconds

            # include_tmax=False pour éviter qu'un échantillon de bord soit dupliqué
            raw_window = eeg.raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

            segment = EEGRecordedDataHelper.update_raw(eeg, raw_window)
            segments.append(segment)

        return segments





    @staticmethod
    def _largest_remainder_split(
        n: int,
        train_ratio: float,
        validate_ratio: float,
        test_ratio: float,
    ) -> tuple[int, int, int]:
        """
        Répartit n éléments entre train / validate / test en respectant au mieux
        les ratios, tout en garantissant que la somme finale vaut exactement n.

        Méthode :
        - on calcule les parts théoriques
        - on prend leur partie entière
        - on distribue le reliquat aux plus grands restes fractionnaires
        """
        raw_counts = {
            "train": n * train_ratio,
            "validate": n * validate_ratio,
            "test": n * test_ratio,
        }

        int_counts = {k: int(v) for k, v in raw_counts.items()}
        assigned = sum(int_counts.values())
        remainder = n - assigned

        fractional_parts = sorted(
            ((k, raw_counts[k] - int_counts[k]) for k in raw_counts),
            key=lambda x: x[1],
            reverse=True,
        )

        for i in range(remainder):
            split_name = fractional_parts[i][0]
            int_counts[split_name] += 1

        return int_counts["train"], int_counts["validate"], int_counts["test"]

    @staticmethod
    def _copy_participant_with_tag(participant: Participant, tag: str) -> Participant:
        """
        Crée une copie logique d'un participant en ne changeant que son tag.
        """
        return Participant(
            id=participant.id,
            gender=participant.gender,
            age=participant.age,
            group=participant.group,
            mmse=participant.mmse,
            tag=tag,
        )

    @staticmethod
    def _copy_eeg_with_tag(eeg: EEGRecordedData, tag: str) -> EEGRecordedData:
        """
        Crée une copie logique d'un EEGRecordedData :
        - même signal brut
        - même fréquence d'échantillonnage
        - même participant, sauf que son tag est remplacé
        """
        tagged_subject = EEGRecordedDataHelper._copy_participant_with_tag(
            eeg.subject,
            tag,
        )

        return EEGRecordedData(
            raw=eeg.raw.copy(),
            sampling_frequency=eeg.sampling_frequency,
            subject=tagged_subject,
        )

    @staticmethod
    def tag(
        eegs: list[EEGRecordedData],
        train_ratio: float = 0.7,
        validate_ratio: float = 0.15,
        test_ratio: float = 0.15,
        *,
        stratify_by: Literal["group", "health_state", "gender", "none"] = "group",
        seed: int = 42,
    ) -> list[EEGRecordedData]:
        """
        Attribue à chaque EEG brut un tag parmi {"train", "validate", "test"}
        en se basant sur les attributs de `eeg.subject`.

        Important
        ---------
        Le tagging doit se faire AVANT le découpage en fenêtres de 1 minute,
        pour éviter qu'un même EEG brut produise des fenêtres réparties
        dans plusieurs splits.

        Paramètres
        ----------
        eegs:
            Liste des EEG bruts à répartir.
        train_ratio / validate_ratio / test_ratio:
            Proportions désirées.
        stratify_by:
            Permet de conserver approximativement la répartition des sujets
            selon :
            - "group"
            - "health_state"
            - "gender"
            - "none"
        seed:
            Graine aléatoire pour rendre le split reproductible.

        Retour
        ------
        Une nouvelle liste de EEGRecordedData taggés.
        """

        total_ratio = train_ratio + validate_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-9:
            raise ValueError(
                f"Les ratios doivent sommer à 1.0. Reçu : {total_ratio}"
            )

        if not eegs:
            return []

        rng = Random(seed)

        # --------------------------------------------------------------
        # Construction des buckets de stratification
        # --------------------------------------------------------------
        if stratify_by == "none":
            groups = {"all": list(eegs)}
        else:
            buckets: dict[str, list[EEGRecordedData]] = defaultdict(list)

            for eeg in eegs:
                subject = eeg.subject

                if stratify_by == "group":
                    key = subject.group
                elif stratify_by == "health_state":
                    key = subject.health_state
                elif stratify_by == "gender":
                    key = subject.gender
                else:
                    raise ValueError(f"stratify_by inconnu : {stratify_by}")

                buckets[key].append(eeg)

            groups = dict(buckets)

        # --------------------------------------------------------------
        # Split dans chaque bucket
        # --------------------------------------------------------------
        tagged_eegs: list[EEGRecordedData] = []

        for _, bucket in groups.items():
            bucket_copy = bucket[:]
            rng.shuffle(bucket_copy)

            n = len(bucket_copy)
            n_train, n_validate, n_test = EEGRecordedDataHelper._largest_remainder_split(
                n=n,
                train_ratio=train_ratio,
                validate_ratio=validate_ratio,
                test_ratio=test_ratio,
            )

            train_part = bucket_copy[:n_train]
            validate_part = bucket_copy[n_train:n_train + n_validate]
            test_part = bucket_copy[n_train + n_validate:n_train + n_validate + n_test]

            tagged_eegs.extend(
                EEGRecordedDataHelper._copy_eeg_with_tag(eeg, "train")
                for eeg in train_part
            )
            tagged_eegs.extend(
                EEGRecordedDataHelper._copy_eeg_with_tag(eeg, "validate")
                for eeg in validate_part
            )
            tagged_eegs.extend(
                EEGRecordedDataHelper._copy_eeg_with_tag(eeg, "test")
                for eeg in test_part
            )

        # Mélange final
        rng.shuffle(tagged_eegs)
        return tagged_eegs