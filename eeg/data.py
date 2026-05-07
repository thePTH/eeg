from __future__ import annotations

import csv
from abc import ABC
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from random import Random
from typing import Callable, Iterator, Literal, Self

import mne
from mne_bids import BIDSPath, read_raw_bids

from eeg.signal import SampledSignal
from participants.definition import Participant, ParticipantFactory
from preprocessing.names import PipelineName
from utils.enum import EnumParser


RawLoader = Callable[[], mne.io.Raw]


class EEGData(ABC):
    """
    Classe racine représentant une donnée EEG.

    Cette classe supporte un mode lazy :
    - soit le Raw est déjà disponible ;
    - soit il est absent mais l'objet sait le reconstruire via `_raw_loader`.

    Attention :
    - `.load()` charge explicitement les données en RAM via `raw.load_data()`.
    - Pour manipuler un Raw sans charger les données, il faut utiliser les méthodes lazy
      du helper, qui n'appellent pas `.load()`.
    """

    def __init__(
        self,
        *,
        raw: mne.io.Raw | None,
        sampling_frequency: float,
        raw_loader: Callable[[], mne.io.Raw] | None = None,
    ):
        self._raw = raw
        self._sampling_frequency = float(sampling_frequency)
        self._raw_loader = raw_loader

    @property
    def sampling_frequency(self) -> float:
        return self._sampling_frequency

    @property
    def is_loaded(self) -> bool:
        return self._raw is not None

    @property
    def can_reload(self) -> bool:
        return self._raw_loader is not None

    @property
    def cache_key(self) -> str:
        return f"{type(self).__name__}:{id(self)}"

    def load(self) -> Self:
        """
        Charge le Raw si nécessaire, puis charge les données en RAM.

        Cette méthode n'est PAS lazy.
        """
        if self._raw is None:
            if self._raw_loader is None:
                raise RuntimeError(
                    "This EEG object cannot be loaded because no raw_loader is available."
                )
            self._raw = self._raw_loader()

        if not self._raw.preload:
            self._raw.load_data(verbose=False)

        return self

    def unload(self) -> None:
        self._raw = None

    @contextmanager
    def loaded(self) -> Iterator[mne.io.Raw]:
        was_loaded = self.is_loaded
        self.load()
        try:
            yield self.raw
        finally:
            if not was_loaded:
                self.unload()

    @property
    def raw(self) -> mne.io.Raw:
        if self._raw is None:
            raise RuntimeError(
                "Raw data is not loaded. Call .load() first or use 'with eeg.loaded()'."
            )
        return self._raw

    @property
    def data(self):
        with self.loaded() as raw:
            return raw.get_data()

    @property
    def signal_names(self) -> list[str]:
        with self.loaded() as raw:
            return list(raw.ch_names)

    @property
    def signals(self) -> list[SampledSignal]:
        return list(self.iter_signals())

    def iter_signals(self) -> Iterator[SampledSignal]:
        with self.loaded() as raw:
            data = raw.get_data()
            ch_names = list(raw.ch_names)

            for channel_index, channel_name in enumerate(ch_names):
                yield SampledSignal(
                    sampling_frequency=self.sampling_frequency,
                    points=data[channel_index],
                    name=channel_name,
                )

    @property
    def info(self):
        with self.loaded() as raw:
            return raw.info

    def _copy_kwargs(self) -> dict:
        return {
            "raw": self._raw.copy() if self._raw is not None else None,
            "sampling_frequency": self.sampling_frequency,
            "raw_loader": self._raw_loader,
        }

    def copy(self) -> Self:
        return type(self)(**self._copy_kwargs())

    def update_raw(self, new_raw: mne.io.Raw, *, copy_raw: bool = False) -> Self:
        kwargs = self._copy_kwargs()
        kwargs["raw"] = new_raw.copy() if copy_raw else new_raw

        # Une fois qu'on a un nouveau Raw transformé,
        # l'ancien loader n'est plus forcément cohérent.
        kwargs["raw_loader"] = None

        return type(self)(**kwargs)

    def plot(self):
        with self.loaded() as raw:
            raw.plot(verbose=False)


class EEGRecordedData(EEGData):
    """
    EEG brut associé à un participant.
    """

    def __init__(
        self,
        *,
        raw: mne.io.Raw | None,
        sampling_frequency: float,
        subject: Participant,
        raw_loader: RawLoader | None = None,
    ):
        super().__init__(
            raw=raw,
            sampling_frequency=sampling_frequency,
            raw_loader=raw_loader,
        )
        self._subject = subject

    @property
    def subject(self) -> Participant:
        return self._subject

    @property
    def cache_key(self) -> str:
        return f"recorded:{self.subject.id}"

    def _copy_kwargs(self) -> dict:
        kwargs = super()._copy_kwargs()
        kwargs["subject"] = self.subject
        return kwargs


class EEGProcessedData(EEGData):
    """
    EEG après preprocessing.
    """

    def __init__(
        self,
        *,
        raw: mne.io.Raw | None,
        source: EEGRecordedData,
        pipeline_name: PipelineName | str,
        raw_loader: RawLoader | None = None,
    ):
        super().__init__(
            raw=raw,
            sampling_frequency=source.sampling_frequency,
            raw_loader=raw_loader,
        )
        self._pipeline_name = EnumParser.parse(pipeline_name, PipelineName)
        self._source = source

    @property
    def pipeline_name(self) -> str:
        return self._pipeline_name.value

    @property
    def source(self) -> EEGRecordedData:
        return self._source

    @property
    def cache_key(self) -> str:
        return f"processed:{self.source.subject.id}:{self.pipeline_name}"

    def _copy_kwargs(self) -> dict:
        kwargs = super()._copy_kwargs()
        kwargs["source"] = self.source
        kwargs["pipeline_name"] = self.pipeline_name
        return kwargs


class EEGRecordedDataProvider:
    """
    Provider chargé de construire les EEG bruts à partir d'un dossier BIDS.
    """

    @staticmethod
    def _extract_subject(row: dict) -> Participant:
        participant_id = row["participant_id"].split("-")[1]
        gender = row["Gender"]
        age = int(row["Age"])
        group = row["Group"]
        mmse = int(row["MMSE"])

        return Participant(
            id=participant_id,
            gender=gender,
            age=age,
            group=group,
            mmse=mmse,
        )

    @staticmethod
    def _build_bids_path(subject: Participant, root: Path) -> BIDSPath:
        return BIDSPath(
            subject=subject.id,
            task="eyesclosed",
            datatype="eeg",
            root=root,
        )

    @staticmethod
    def _make_raw_loader(subject: Participant, root: Path) -> RawLoader:
        """
        Construit un loader lazy.

        Important :
        - ne pas appeler raw.load_data()
        - le Raw retourné reste en preload=False
        """

        def loader() -> mne.io.Raw:
            bids_path = EEGRecordedDataProvider._build_bids_path(subject, root)
            raw = read_raw_bids(bids_path=bids_path, verbose=False)

            montage = mne.channels.make_standard_montage("standard_1020")
            raw.set_montage(montage, verbose=False)

            return raw

        return loader

    @staticmethod
    def _extract_recorded_eeg(
        subject: Participant,
        root: Path,
        load_data: bool = True,
    ) -> EEGRecordedData:
        bids_path = EEGRecordedDataProvider._build_bids_path(subject, root)
        raw_preview: mne.io.Raw = read_raw_bids(bids_path=bids_path, verbose=False)

        montage = mne.channels.make_standard_montage("standard_1020")
        raw_preview.set_montage(montage, verbose=False)

        sampling_frequency = float(raw_preview.info["sfreq"])
        raw_loader = EEGRecordedDataProvider._make_raw_loader(subject, root)

        if load_data:
            raw_preview.load_data(verbose=False)
            raw = raw_preview
        else:
            raw = raw_preview

        return EEGRecordedData(
            raw=raw,
            sampling_frequency=sampling_frequency,
            subject=subject,
            raw_loader=raw_loader,
        )

    @staticmethod
    def build(data_file_path: str, load_data: bool = True) -> list[EEGRecordedData]:
        recordings: list[EEGRecordedData] = []
        root = Path(data_file_path)

        with open(root / "participants.tsv", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for row in reader:
                subject = EEGRecordedDataProvider._extract_subject(row)

                recorded_eeg = EEGRecordedDataProvider._extract_recorded_eeg(
                    subject=subject,
                    root=root,
                    load_data=load_data,
                )

                recordings.append(recorded_eeg)

        return recordings


class EEGRecordedDataHelper:
    """
    Helper métier pour manipuler des EEG bruts.
    """

    @staticmethod
    def update_raw(eeg: EEGData, new_raw: mne.io.Raw) -> Self:
        return eeg.update_raw(new_raw, copy_raw=False)

    @staticmethod
    def iter_split(
        eeg: EEGRecordedData,
        t_start: int = 10,
        window_seconds: int = 60,
    ) -> Iterator[EEGRecordedData]:
        """
        Découpe classique.

        Attention :
        cette méthode charge le Raw en RAM car elle utilise `eeg.loaded()`.
        """
        with eeg.loaded() as raw:
            total_duration = float(raw.times[-1])

            if t_start >= total_duration:
                return

            usable_duration = total_duration - t_start
            n_full_windows = int(usable_duration // window_seconds)

            for i in range(n_full_windows):
                tmin = t_start + i * window_seconds
                tmax = tmin + window_seconds

                raw_window = raw.copy().crop(
                    tmin=tmin,
                    tmax=tmax,
                    include_tmax=False,
                )

                yield eeg.update_raw(raw_window)

    @staticmethod
    def split(
        eeg: EEGRecordedData,
        t_start: int = 10,
        window_seconds: int = 60,
    ) -> list[EEGRecordedData]:
        return list(
            EEGRecordedDataHelper.iter_split(
                eeg=eeg,
                t_start=t_start,
                window_seconds=window_seconds,
            )
        )

    @staticmethod
    def _get_raw_without_loading(eeg: EEGRecordedData) -> mne.io.Raw:
        """
        Récupère un Raw sans appeler eeg.load().

        Si eeg._raw existe, on l'utilise directement.
        Sinon, on appelle le raw_loader, qui doit lui-même retourner un Raw non chargé.
        """
        if eeg._raw is not None:
            raw = eeg._raw
        elif eeg._raw_loader is not None:
            raw = eeg._raw_loader()
        else:
            raise RuntimeError(
                "Impossible de récupérer un Raw : aucun Raw ni raw_loader disponible."
            )

        if raw.preload:
            raise RuntimeError(
                "Le Raw est déjà chargé en mémoire. "
                "Pour un découpage lazy, il faut un Raw avec preload=False."
            )

        return raw

    @staticmethod
    def iter_split_lazy(
        eeg: EEGRecordedData,
        t_start: int = 10,
        window_seconds: int = 60,
    ) -> Iterator[EEGRecordedData]:
        """
        Découpe un EEGRecordedData en fenêtres sans charger les données en RAM.

        Cette méthode :
        - n'appelle jamais eeg.load()
        - n'appelle jamais raw.load_data()
        - conserve preload=False
        - découpe en contrôlant les samples via la fréquence d'échantillonnage
        """
        raw = EEGRecordedDataHelper._get_raw_without_loading(eeg)

        sfreq = float(raw.info["sfreq"])

        start_sample = int(t_start * sfreq)
        window_samples = int(window_seconds * sfreq)

        if window_samples <= 0:
            raise ValueError("window_seconds doit être strictement positif.")

        n_times = raw.n_times

        if start_sample >= n_times:
            return

        n_full_windows = (n_times - start_sample) // window_samples

        for i in range(n_full_windows):
            sample_min = start_sample + i * window_samples
            sample_max = sample_min + window_samples

            tmin = sample_min / sfreq
            tmax = sample_max / sfreq

            raw_window = raw.copy().crop(
                tmin=tmin,
                tmax=tmax,
                include_tmax=False,
            )

            yield eeg.update_raw(raw_window)

    @staticmethod
    def split_lazy(
        eeg: EEGRecordedData,
        t_start: int = 10,
        window_seconds: int = 60,
    ) -> list[EEGRecordedData]:
        """
        Version liste de iter_split_lazy.
        """
        return list(
            EEGRecordedDataHelper.iter_split_lazy(
                eeg=eeg,
                t_start=t_start,
                window_seconds=window_seconds,
            )
        )

    @staticmethod
    def _largest_remainder_split(
        n: int,
        train_ratio: float,
        validate_ratio: float,
        test_ratio: float,
    ) -> tuple[int, int, int]:
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
        tagged_subject = EEGRecordedDataHelper._copy_participant_with_tag(
            eeg.subject,
            tag,
        )

        return EEGRecordedData(
            raw=eeg._raw.copy() if eeg.is_loaded else None,
            sampling_frequency=eeg.sampling_frequency,
            subject=tagged_subject,
            raw_loader=eeg._raw_loader,
        )

    

import json




RawLoader = Callable[[], mne.io.Raw]





class EEGProcessedDataIO:
    """
    I/O optimisé pour EEGProcessedData.

    Structure créée
    ----------------
    path/
    └── sub-<id>-rec-XX/
        ├── raw.fif.gz
        └── metadata.json

    XX est automatiquement incrémenté selon les enregistrements déjà présents.

    Important :
    - le nom du dossier suit exactement la même logique que
      SingleParticipantProcessedFeatureDatasetIO ;
    - cela permet de relier un EEGProcessedData exporté avec le
      SingleParticipantProcessedFeatureDataset correspondant.
    """

    RAW_FILENAME = "raw.fif.gz"
    METADATA_FILENAME = "metadata.json"

    @staticmethod
    def _build_export_folder(
        *,
        root: Path,
        subject_id: str,
    ) -> tuple[Path, str]:
        subject_prefix = f"sub-{subject_id}-rec-"

        existing_indices = []

        for folder in root.iterdir():
            if folder.is_dir() and folder.name.startswith(subject_prefix):
                suffix = folder.name.replace(subject_prefix, "")

                if suffix.isdigit():
                    existing_indices.append(int(suffix))

        next_index = 1 if not existing_indices else max(existing_indices) + 1
        recording_key = f"{next_index:02d}"

        export_folder = root / f"{subject_prefix}{recording_key}"

        return export_folder, recording_key

    @staticmethod
    def export(
        eeg: EEGProcessedData,
        path: str | Path,
        *,
        overwrite: bool = False,
    ) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        subject_id = eeg.source.subject.id

        export_folder, recording_key = EEGProcessedDataIO._build_export_folder(
            root=path,
            subject_id=subject_id,
        )

        if export_folder.exists() and not overwrite:
            raise FileExistsError(f"Folder already exists: {export_folder}")

        export_folder.mkdir(parents=True, exist_ok=True)

        raw_path = export_folder / EEGProcessedDataIO.RAW_FILENAME
        metadata_path = export_folder / EEGProcessedDataIO.METADATA_FILENAME

        with eeg.loaded() as raw:
            raw.save(
                raw_path,
                overwrite=overwrite,
                fmt="single",
                verbose=False,
            )

        metadata = {
            "subject_dico": eeg.source.subject.to_dict(),
            "pipeline_name": eeg.pipeline_name,
            "sampling_frequency": eeg.sampling_frequency,
            "recording_key": recording_key,
        }

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return export_folder

    @staticmethod
    def load(path: str | Path) -> EEGProcessedData:
        path = Path(path)

        raw_path = path / EEGProcessedDataIO.RAW_FILENAME
        metadata_path = path / EEGProcessedDataIO.METADATA_FILENAME

        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw file: {raw_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        source = EEGRecordedData(
            raw=None,
            sampling_frequency=float(metadata["sampling_frequency"]),
            subject=ParticipantFactory.build(metadata["subject_dico"]),
            raw_loader=None,
        )

        def raw_loader() -> mne.io.Raw:
            return mne.io.read_raw_fif(
                raw_path,
                preload=False,
                verbose=False,
            )

        return EEGProcessedData(
            raw=None,
            source=source,
            pipeline_name=metadata["pipeline_name"],
            raw_loader=raw_loader,
        )