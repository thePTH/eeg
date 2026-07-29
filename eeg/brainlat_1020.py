from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.optimize import linear_sum_assignment

from eeg.data import EEGRecordedData, RawLoader
from participants.definition import Participant


class BrainLat1020Error(RuntimeError):
    """Raised when a BrainLat recording cannot be converted to 19 channels."""


class BrainLat1020EEGRecordedDataProvider:
    """
    Load BrainLat AD and healthy-control EEGs as 19-channel recordings.

    The provider intentionally implements a small, dataset-specific contract:

    - only ``1_AD`` and ``5_HC`` are scanned;
    - BrainLat ``HC`` participants are exposed as project group ``CN``;
    - acquisition country/site is ignored;
    - each recording is reduced to the canonical 19-channel 10-20 montage;
    - loading remains lazy and returns ``list[EEGRecordedData]``.

    When canonical channel names are not already present (BrainLat commonly
    uses BioSemi labels such as A1-D32), the closest unique channels are chosen
    from the coordinates embedded in the EEGLAB file. They are then renamed to
    the canonical 10-20 names and ordered consistently.
    """

    CHANNELS_1020: tuple[str, ...] = (
        "Fp1", "Fp2",
        "F7", "F3", "Fz", "F4", "F8",
        "T7", "C3", "Cz", "C4", "T8",
        "P7", "P3", "Pz", "P4", "P8",
        "O1", "O2",
    )

    GROUP_DIRECTORIES: dict[str, str] = {
        "AD": "1_AD",
        "CN": "5_HC",
    }

    DEMOGRAPHICS_FILENAMES: dict[str, str] = {
        "AD": "Demographics_AD_EEG_data.csv",
        "CN": "Demographics_HC_EEG_data.csv",
    }

    _ID_COLUMNS = ("ideeg", "participantid", "subjectid", "subject", "id")
    _AGE_COLUMNS = ("age", "ageyears")
    _SEX_COLUMNS = ("sex", "gender")
    _MISSING = {"", "na", "n/a", "nan", "none", "null", "unknown"}

    @staticmethod
    def _normalized_key(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())

    @classmethod
    def _normalized_row(cls, row: Mapping[str, Any]) -> dict[str, Any]:
        return {
            cls._normalized_key(key): value
            for key, value in row.items()
            if key is not None
        }

    @classmethod
    def _clean_string(cls, value: Any) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return None if cleaned.lower() in cls._MISSING else cleaned

    @classmethod
    def _clean_int(cls, value: Any) -> int | None:
        cleaned = cls._clean_string(value)
        if cleaned is None:
            return None
        try:
            return int(round(float(cleaned.replace(",", "."))))
        except ValueError:
            return None

    @classmethod
    def _first_value(
        cls,
        row: Mapping[str, Any],
        candidates: tuple[str, ...],
    ) -> Any:
        for candidate in candidates:
            if candidate in row:
                return row[candidate]
        return None

    @classmethod
    def _normalize_subject_id(cls, value: Any) -> str | None:
        subject_id = cls._clean_string(value)
        if subject_id is None:
            return None
        subject_id = subject_id.strip()
        if subject_id.lower().startswith("sub-"):
            subject_id = subject_id[4:]
        return subject_id or None

    @classmethod
    def _read_demographics(cls, path: Path) -> dict[str, dict[str, Any]]:
        """Read the optional demographics file and index it by subject ID."""
        if not path.exists():
            return {}

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                with path.open("r", encoding=encoding, newline="") as stream:
                    sample = stream.read(4096)
                    stream.seek(0)
                    try:
                        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
                    except csv.Error:
                        delimiter = ","

                    indexed: dict[str, dict[str, Any]] = {}
                    for raw_row in csv.DictReader(stream, delimiter=delimiter):
                        row = cls._normalized_row(raw_row)
                        subject_id = cls._normalize_subject_id(
                            cls._first_value(row, cls._ID_COLUMNS)
                        )
                        if subject_id is not None:
                            indexed[subject_id] = row
                    return indexed
            except UnicodeDecodeError:
                continue

        raise BrainLat1020Error(f"Unable to decode demographics file: {path}")

    @classmethod
    def _subject_id_from_path(cls, eeg_path: Path) -> str:
        for parent in (eeg_path.parent, *eeg_path.parents):
            if parent.name.lower().startswith("sub-"):
                subject_id = cls._normalize_subject_id(parent.name)
                if subject_id is not None:
                    return subject_id
        raise BrainLat1020Error(
            f"No sub-<id> directory found above EEGLAB file: {eeg_path}"
        )

    @classmethod
    def _parse_gender(cls, value: Any) -> str | None:
        cleaned = cls._clean_string(value)
        if cleaned is None:
            return None
        normalized = cleaned.strip().upper()
        aliases = {
            "M": "M", "MALE": "M", "MAN": "M", "1": "M", "1.0": "M",
            "F": "F", "FEMALE": "F", "WOMAN": "F", "0": "F", "0.0": "F",
        }
        return aliases.get(normalized)

    @classmethod
    def _build_participant(
        cls,
        *,
        subject_id: str,
        group: str,
        demographics: Mapping[str, Any] | None,
        eeg_path: Path,
    ) -> Participant:
        row = demographics or {}
        return Participant(
            id=subject_id,
            gender=cls._parse_gender(cls._first_value(row, cls._SEX_COLUMNS)),
            age=cls._clean_int(cls._first_value(row, cls._AGE_COLUMNS)),
            group=group,
            metadata={
                "dataset": "BrainLat",
                "eeg_file": str(eeg_path),
                "original_group": "HC" if group == "CN" else "AD",
            },
        )

    @staticmethod
    def _read_raw_eeglab(eeg_path: Path, *, preload: bool) -> mne.io.Raw:
        errors: list[Exception] = []
        for kwargs in ({}, {"uint16_codec": "latin1"}, {"uint16_codec": "utf-8"}):
            try:
                return mne.io.read_raw_eeglab(
                    eeg_path,
                    preload=preload,
                    verbose=False,
                    **kwargs,
                )
            except Exception as error:  # MNE may raise several backend errors.
                errors.append(error)

        details = "\n".join(
            f"- {type(error).__name__}: {error}" for error in errors
        )
        raise BrainLat1020Error(
            f"Unable to read BrainLat EEGLAB file {eeg_path}:\n{details}"
        )

    @staticmethod
    def _unit_vectors(positions: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(positions, axis=1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise BrainLat1020Error("Missing or invalid EEG channel coordinates.")
        return positions / norms

    @classmethod
    def _direct_channel_mapping(cls, raw: mne.io.Raw) -> dict[str, str] | None:
        """Return canonical -> source mapping when names already identify all channels."""
        aliases: dict[str, str] = {}
        for source_name in raw.ch_names:
            key = cls._normalized_key(source_name)
            key = re.sub(r"^(eeg|channel|ch)", "", key)
            aliases.setdefault(key, source_name)

        mapping: dict[str, str] = {}
        for canonical_name in cls.CHANNELS_1020:
            source_name = aliases.get(cls._normalized_key(canonical_name))
            if source_name is None:
                return None
            mapping[canonical_name] = source_name
        return mapping

    @classmethod
    def _coordinate_channel_mapping(cls, raw: mne.io.Raw) -> dict[str, str]:
        """Match 19 standard positions to 19 unique source channels globally."""
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(eeg_picks) < len(cls.CHANNELS_1020):
            raise BrainLat1020Error(
                f"Recording contains only {len(eeg_picks)} EEG channels; 19 required."
            )

        source_names: list[str] = []
        source_positions: list[np.ndarray] = []
        for pick in eeg_picks:
            position = np.asarray(raw.info["chs"][pick]["loc"][:3], dtype=float)
            if np.all(np.isfinite(position)) and np.linalg.norm(position) > 0:
                source_names.append(raw.ch_names[pick])
                source_positions.append(position)

        if len(source_positions) < len(cls.CHANNELS_1020):
            raise BrainLat1020Error(
                "Fewer than 19 EEG channels have usable coordinates in the EEGLAB file."
            )

        standard_montage = mne.channels.make_standard_montage("standard_1020")
        standard_positions = standard_montage.get_positions()["ch_pos"]
        target_positions = np.asarray(
            [standard_positions[name] for name in cls.CHANNELS_1020],
            dtype=float,
        )

        source_vectors = cls._unit_vectors(np.asarray(source_positions))
        target_vectors = cls._unit_vectors(target_positions)

        # Angular distance is robust to different coordinate scales.
        similarities = np.clip(target_vectors @ source_vectors.T, -1.0, 1.0)
        angular_cost = np.arccos(similarities)
        target_indices, source_indices = linear_sum_assignment(angular_cost)

        mapping = {
            cls.CHANNELS_1020[target_index]: source_names[source_index]
            for target_index, source_index in zip(target_indices, source_indices)
        }
        if len(mapping) != len(cls.CHANNELS_1020):
            raise BrainLat1020Error("Could not produce a complete 19-channel mapping.")
        return mapping

    @classmethod
    def _keep_19_channels(cls, raw: mne.io.Raw) -> mne.io.Raw:
        """Select, rename and order the canonical 19-channel 10-20 montage."""
        mapping = cls._direct_channel_mapping(raw)
        if mapping is None:
            mapping = cls._coordinate_channel_mapping(raw)

        selected_sources = [mapping[name] for name in cls.CHANNELS_1020]
        raw.pick(selected_sources)

        rename_mapping = {
            source_name: canonical_name
            for canonical_name, source_name in mapping.items()
            if source_name != canonical_name
        }
        if rename_mapping:
            raw.rename_channels(rename_mapping)

        raw.reorder_channels(list(cls.CHANNELS_1020))
        raw.set_channel_types(
            {name: "eeg" for name in cls.CHANNELS_1020},
            on_unit_change="ignore",
            verbose=False,
        )
        raw.set_montage(
            mne.channels.make_standard_montage("standard_1020"),
            on_missing="raise",
            verbose=False,
        )
        return raw

    @classmethod
    def _load_and_prepare(cls, eeg_path: Path, *, preload: bool) -> mne.io.Raw:
        raw = cls._read_raw_eeglab(eeg_path, preload=preload)
        return cls._keep_19_channels(raw)

    @classmethod
    def _make_raw_loader(cls, eeg_path: Path) -> RawLoader:
        def loader() -> mne.io.Raw:
            return cls._load_and_prepare(eeg_path, preload=False)
        return loader

    @classmethod
    def _build_recording(
        cls,
        *,
        eeg_path: Path,
        group: str,
        demographics: Mapping[str, Any] | None,
        load_data: bool,
    ) -> EEGRecordedData:
        subject_id = cls._subject_id_from_path(eeg_path)
        participant = cls._build_participant(
            subject_id=subject_id,
            group=group,
            demographics=demographics,
            eeg_path=eeg_path,
        )
        raw = cls._load_and_prepare(eeg_path, preload=load_data)
        return EEGRecordedData(
            raw=raw,
            sampling_frequency=float(raw.info["sfreq"]),
            subject=participant,
            raw_loader=cls._make_raw_loader(eeg_path),
        )

    @classmethod
    def build(
        cls,
        data_file_path: str | Path,
        *,
        load_data: bool = False,
        skip_invalid: bool = False,
    ) -> list[EEGRecordedData]:
        """
        Return all available BrainLat AD and HC recordings.

        ``data_file_path`` must point to the BrainLat ``EEG data`` directory
        containing ``1_AD`` and ``5_HC``. The country subdirectories are
        traversed recursively but are otherwise ignored.
        """
        root = Path(data_file_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"BrainLat directory does not exist: {root}")

        recordings: list[EEGRecordedData] = []
        seen_subjects: set[tuple[str, str]] = set()

        for group, directory_name in cls.GROUP_DIRECTORIES.items():
            group_root = root / directory_name
            if not group_root.exists():
                raise FileNotFoundError(f"Missing BrainLat group directory: {group_root}")

            demographics = cls._read_demographics(
                group_root / cls.DEMOGRAPHICS_FILENAMES[group]
            )
            eeg_paths = sorted(path for path in group_root.rglob("*.set") if path.is_file())

            for eeg_path in eeg_paths:
                try:
                    subject_id = cls._subject_id_from_path(eeg_path)
                    key = (group, subject_id)
                    if key in seen_subjects:
                        raise BrainLat1020Error(
                            f"Multiple EEGLAB files found for {group} sub-{subject_id}."
                        )

                    recording = cls._build_recording(
                        eeg_path=eeg_path,
                        group=group,
                        demographics=demographics.get(subject_id),
                        load_data=load_data,
                    )
                    recordings.append(recording)
                    seen_subjects.add(key)
                except (BrainLat1020Error, FileNotFoundError, ValueError) as error:
                    if not skip_invalid:
                        raise
                    warnings.warn(f"Skipping {eeg_path}: {error}", RuntimeWarning)

        return recordings
