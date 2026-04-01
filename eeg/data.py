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
    def _extract_recorded_eeg(subject:Participant, root:Path):
        bids_path = BIDSPath(
            subject=subject.id,
            task="eyesclosed",
            datatype="eeg",
            root=root
        )
        raw : mne.io.Raw = read_raw_bids(bids_path=bids_path, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage, verbose=False)

        raw.load_data(verbose=False)

        return EEGRecordedData(raw=raw, sampling_frequency=raw.info['sfreq'], subject=subject)




    @staticmethod
    def build(data_file_path:str) -> list[EEGRecordedData]:
        recordings = []
        root = Path(data_file_path)
        

        with open(root/"participants.tsv", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                subject = EEGRecordedDataProvider._extract_subject(row)
                recorded_eeg = EEGRecordedDataProvider._extract_recorded_eeg(subject, root)
                recordings.append(recorded_eeg)

        return recordings
                


    

