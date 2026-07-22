from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mne

from eeg.data import EEGRecordedData, RawLoader
from participants.definition import Participant


class BrainLatDatasetError(RuntimeError):
    """Raised when the BrainLat dataset is missing or inconsistent."""


class BrainLatEEGRecordedDataProvider:
    """
    Provider used to build EEGRecordedData objects from BrainLat.

    Expected dataset structure::

        EEG data/
        ├── 1_AD/
        │   ├── AR/
        │   │   └── sub-30001/
        │   │       └── eeg/
        │   │           ├── recording.set
        │   │           └── recording.fdt
        │   ├── CL/
        │   ├── Records_AD_EEG_data.csv
        │   ├── Demographics_AD_EEG_data.csv
        │   └── Cognition_AD_EEG_data.csv
        │
        └── 5_HC/
            ├── AR/
            ├── CL/
            ├── Records_HC_EEG_data.csv
            ├── Demographics_HC_EEG_data.csv
            └── Cognition_HC_EEG_data.csv

    Internal diagnostic groups follow the existing project conventions:

    - ``AD`` for Alzheimer disease;
    - ``CN`` for cognitively normal participants.

    BrainLat labels or directory names using ``HC`` are converted to ``CN``.

    Notes
    -----
    BrainLat contains high-density EEG recordings. Channel names and channel
    coordinates contained in the EEGLAB files are preserved. This provider
    deliberately does not apply MNE's ``standard_1020`` montage.
    """

    GROUP_CONFIG: dict[str, dict[str, str]] = {
        "AD": {
            "directory": "1_AD",
            "records": "Records_AD_EEG_data.csv",
            "demographics": "Demographics_AD_EEG_data.csv",
            "cognition": "Cognition_AD_EEG_data.csv",
        },
        "CN": {
            "directory": "5_HC",
            "records": "Records_HC_EEG_data.csv",
            "demographics": "Demographics_HC_EEG_data.csv",
            "cognition": "Cognition_HC_EEG_data.csv",
        },
    }

    DIAGNOSIS_ALIASES: dict[str, str] = {
        # Alzheimer disease
        "AD": "AD",
        "ALZHEIMER": "AD",
        "ALZHEIMERS": "AD",
        "ALZHEIMER'S": "AD",
        "ALZHEIMER DISEASE": "AD",
        "ALZHEIMER'S DISEASE": "AD",
        "ALZHEIMERS DISEASE": "AD",

        # Cognitively normal / healthy controls
        "CN": "CN",
        "HC": "CN",
        "HEALTHY": "CN",
        "CONTROL": "CN",
        "CONTROLS": "CN",
        "HEALTHY CONTROL": "CN",
        "HEALTHY CONTROLS": "CN",
        "COGNITIVELY NORMAL": "CN",
        "COGNITIVE NORMAL": "CN",
        "NORMAL CONTROL": "CN",
        "NORMAL CONTROLS": "CN",
    }

    MISSING_VALUES = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "unknown",
        "missing",
    }

    ID_COLUMN_CANDIDATES = (
        "ideeg",
        "participantid",
        "subjectid",
        "id",
    )

    EEG_AVAILABLE_VALUES = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "available",
    }

    AUXILIARY_CHANNEL_PATTERNS: dict[str, tuple[str, ...]] = {
        "eog": (
            "EOG",
            "HEOG",
            "VEOG",
            "LEOG",
            "REOG",
        ),
        "ecg": (
            "ECG",
            "EKG",
        ),
        "emg": (
            "EMG",
        ),
    }

    # -------------------------------------------------------------------------
    # General metadata utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _normalize_column_name(name: Any) -> str:
        """
        Normalize a table column name.

        Examples
        --------
        ``id EEG`` becomes ``ideeg``.
        ``id_EEG`` becomes ``ideeg``.
        ``moca_total`` becomes ``mocatotal``.
        """
        return re.sub(
            pattern=r"[^a-z0-9]+",
            repl="",
            string=str(name).strip().lower(),
        )

    @classmethod
    def _normalize_row(
        cls,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return a metadata row with normalized column names."""
        return {
            cls._normalize_column_name(key): value
            for key, value in row.items()
            if key is not None
        }

    @classmethod
    def _clean_string(
        cls,
        value: Any,
    ) -> str | None:
        """Normalize an optional string value."""
        if value is None:
            return None

        normalized = str(value).strip()

        if normalized.lower() in cls.MISSING_VALUES:
            return None

        return normalized

    @classmethod
    def _clean_float(
        cls,
        value: Any,
    ) -> float | None:
        """Parse an optional floating-point value."""
        normalized = cls._clean_string(value)

        if normalized is None:
            return None

        normalized = normalized.replace(",", ".")

        try:
            number = float(normalized)
        except (TypeError, ValueError):
            return None

        if math.isnan(number):
            return None

        return number

    @classmethod
    def _clean_int(
        cls,
        value: Any,
    ) -> int | None:
        """Parse an optional integer value."""
        number = cls._clean_float(value)

        if number is None:
            return None

        return int(round(number))

    # -------------------------------------------------------------------------
    # Diagnosis handling
    # -------------------------------------------------------------------------

    @classmethod
    def _normalize_diagnosis(
        cls,
        value: Any,
    ) -> str | None:
        """
        Normalize a BrainLat diagnosis to the project conventions.

        Examples
        --------
        ``AD`` becomes ``AD``.
        ``CN`` becomes ``CN``.
        ``HC`` becomes ``CN``.
        ``Healthy Control`` becomes ``CN``.
        """
        diagnosis = cls._clean_string(value)

        if diagnosis is None:
            return None

        normalized = diagnosis.upper().strip()
        normalized = normalized.replace("_", " ")
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"\s+", " ", normalized)

        return cls.DIAGNOSIS_ALIASES.get(
            normalized,
            normalized,
        )

    @classmethod
    def _validate_metadata_diagnoses(
        cls,
        *,
        participant_id: str,
        expected_group: str,
        record_row: Mapping[str, Any],
        demographic_row: Mapping[str, Any],
        cognition_row: Mapping[str, Any] | None,
    ) -> dict[str, str | None]:
        """
        Validate diagnoses from Records, Demographics and Cognition.

        Values are normalized before comparison, so ``HC`` and ``CN`` are
        treated as the same internal group: ``CN``.
        """
        original_diagnoses = {
            "records": cls._clean_string(
                record_row.get("diagnosis")
            ),
            "demographics": cls._clean_string(
                demographic_row.get("diagnosis")
            ),
            "cognition": (
                cls._clean_string(
                    cognition_row.get("diagnosis")
                )
                if cognition_row is not None
                else None
            ),
        }

        normalized_expected_group = cls._normalize_diagnosis(
            expected_group
        )

        if normalized_expected_group not in cls.GROUP_CONFIG:
            raise BrainLatDatasetError(
                f"Invalid expected group for sub-{participant_id}: "
                f"{expected_group!r}."
            )

        for source_name, original_diagnosis in original_diagnoses.items():
            normalized_diagnosis = cls._normalize_diagnosis(
                original_diagnosis
            )

            if normalized_diagnosis is None:
                continue

            if normalized_diagnosis != normalized_expected_group:
                raise BrainLatDatasetError(
                    f"Inconsistent diagnosis for sub-{participant_id} "
                    f"in {source_name}: expected "
                    f"{normalized_expected_group}, got "
                    f"{normalized_diagnosis} "
                    f"(original value={original_diagnosis!r})."
                )

        return original_diagnoses

    # -------------------------------------------------------------------------
    # Metadata table loading
    # -------------------------------------------------------------------------

    @staticmethod
    def _detect_delimiter(sample: str) -> str:
        """Detect comma, semicolon or tab delimiters."""
        try:
            dialect = csv.Sniffer().sniff(
                sample,
                delimiters=",;\t",
            )
            return dialect.delimiter
        except csv.Error:
            return ","

    @classmethod
    def _read_table(
        cls,
        path: Path,
    ) -> list[dict[str, Any]]:
        """Read a BrainLat CSV or TSV metadata table."""
        if not path.exists():
            raise FileNotFoundError(
                f"Missing BrainLat metadata file: {path}"
            )

        decode_error: UnicodeDecodeError | None = None

        for encoding in (
            "utf-8-sig",
            "utf-8",
            "cp1252",
            "latin-1",
        ):
            try:
                with path.open(
                    "r",
                    encoding=encoding,
                    newline="",
                ) as file:
                    sample = file.read(8192)
                    file.seek(0)

                    delimiter = (
                        "\t"
                        if path.suffix.lower() == ".tsv"
                        else cls._detect_delimiter(sample)
                    )

                    reader = csv.DictReader(
                        file,
                        delimiter=delimiter,
                    )

                    if reader.fieldnames is None:
                        raise BrainLatDatasetError(
                            f"No header found in metadata file: {path}"
                        )

                    return list(reader)

            except UnicodeDecodeError as error:
                decode_error = error

        raise BrainLatDatasetError(
            f"Unable to decode BrainLat metadata file: {path}"
        ) from decode_error

    @classmethod
    def _extract_subject_id(
        cls,
        row: Mapping[str, Any],
    ) -> str:
        """Extract the subject identifier from a metadata row."""
        normalized_row = cls._normalize_row(row)

        for candidate in cls.ID_COLUMN_CANDIDATES:
            subject_id = cls._clean_string(
                normalized_row.get(candidate)
            )

            if subject_id is None:
                continue

            subject_id = re.sub(
                pattern=r"^sub-",
                repl="",
                string=subject_id,
                flags=re.IGNORECASE,
            )

            if subject_id:
                return subject_id

        raise BrainLatDatasetError(
            "Unable to find a subject identifier in metadata row: "
            f"{dict(row)}"
        )

    @classmethod
    def _index_rows_by_subject(
        cls,
        rows: list[dict[str, Any]],
        *,
        source_name: str,
    ) -> dict[str, dict[str, Any]]:
        """Index normalized metadata rows by subject identifier."""
        indexed_rows: dict[str, dict[str, Any]] = {}

        for row in rows:
            subject_id = cls._extract_subject_id(row)

            if subject_id in indexed_rows:
                raise BrainLatDatasetError(
                    f"Duplicate subject sub-{subject_id} in "
                    f"{source_name}."
                )

            indexed_rows[subject_id] = cls._normalize_row(row)

        return indexed_rows

    # -------------------------------------------------------------------------
    # Participant construction
    # -------------------------------------------------------------------------

    @classmethod
    def _parse_gender(
        cls,
        value: Any,
    ) -> str | None:
        """
        Parse the BrainLat sex coding.

        Based on the BrainLat dictionary:

        - ``0`` means female;
        - ``1`` means male.
        """
        normalized = cls._clean_string(value)

        if normalized is None:
            return None

        normalized_lower = normalized.lower()

        if normalized_lower in {
            "0",
            "0.0",
            "f",
            "female",
            "woman",
            "femme",
        }:
            return "F"

        if normalized_lower in {
            "1",
            "1.0",
            "m",
            "male",
            "man",
            "homme",
        }:
            return "M"

        return normalized.upper()

    @classmethod
    def _extract_site(
        cls,
        *rows: Mapping[str, Any] | None,
    ) -> str | None:
        """Extract the acquisition country code AR or CL."""
        for row in rows:
            if row is None:
                continue

            path_value = cls._clean_string(
                row.get("path")
            )

            if path_value is None:
                continue

            for path_part in re.split(
                r"[\\/]",
                path_value,
            ):
                candidate = path_part.strip().upper()

                if candidate in {"AR", "CL"}:
                    return candidate

        return None

    @classmethod
    def _is_eeg_available(
        cls,
        record_row: Mapping[str, Any],
    ) -> bool:
        """Return whether EEG is marked as available in Records."""
        value = record_row.get("eeg")

        if value is None:
            # The real file existence is checked afterward.
            return True

        normalized = cls._clean_string(value)

        if normalized is None:
            return False

        return normalized.lower() in cls.EEG_AVAILABLE_VALUES

    @classmethod
    def _extract_cognitive_metadata(
        cls,
        cognition_row: Mapping[str, Any] | None,
    ) -> dict[str, float]:
        """Extract all available numeric cognitive variables."""
        if cognition_row is None:
            return {}

        excluded_columns = {
            "path",
            "ideeg",
            "id",
            "participantid",
            "subjectid",
            "diagnosis",
        }

        cognition: dict[str, float] = {}

        for key, value in cognition_row.items():
            if key in excluded_columns:
                continue

            numeric_value = cls._clean_float(value)

            if numeric_value is not None:
                cognition[key] = numeric_value

        return cognition

    @classmethod
    def _extract_participant(
        cls,
        *,
        subject_id: str,
        expected_group: str,
        record_row: Mapping[str, Any],
        demographic_row: Mapping[str, Any],
        cognition_row: Mapping[str, Any] | None,
    ) -> Participant:
        """Build a project Participant from BrainLat metadata."""
        normalized_group = cls._normalize_diagnosis(
            expected_group
        )

        if normalized_group not in cls.GROUP_CONFIG:
            raise BrainLatDatasetError(
                f"Unsupported group for sub-{subject_id}: "
                f"{expected_group!r}."
            )

        original_diagnoses = cls._validate_metadata_diagnoses(
            participant_id=subject_id,
            expected_group=normalized_group,
            record_row=record_row,
            demographic_row=demographic_row,
            cognition_row=cognition_row,
        )

        site = cls._extract_site(
            record_row,
            demographic_row,
            cognition_row,
        )

        cognition_metadata = cls._extract_cognitive_metadata(
            cognition_row
        )

        moca = cognition_metadata.get("mocatotal")

        metadata = {
            "dataset": "BrainLat",
            "dataset_format": "EEGLAB",
            "diagnosis": normalized_group,
            "original_diagnosis": original_diagnoses,
            "site": site,
            "country": cls._country_from_site(site),
            "years_education": cls._clean_float(
                demographic_row.get("yearseducation")
            ),
            "laterality": cls._clean_int(
                demographic_row.get("laterality")
            ),
            "cognition": cognition_metadata,
        }

        return Participant(
            id=subject_id,
            gender=cls._parse_gender(
                demographic_row.get("sex")
            ),
            age=cls._clean_int(
                demographic_row.get("age")
            ),
            group=normalized_group,
            mmse=None,
            moca=moca,
            metadata=metadata,
        )

    @staticmethod
    def _country_from_site(
        site: str | None,
    ) -> str | None:
        """Convert the BrainLat folder code to a country name."""
        country_by_site = {
            "AR": "Argentina",
            "CL": "Chile",
        }

        return country_by_site.get(site)

    # -------------------------------------------------------------------------
    # EEG file discovery
    # -------------------------------------------------------------------------

    @classmethod
    def _find_eeg_files(
        cls,
        *,
        group_root: Path,
        subject_id: str,
        site: str | None,
    ) -> list[Path]:
        """Find EEGLAB SET files associated with a participant."""
        subject_folder = f"sub-{subject_id}"

        search_roots: list[Path] = []

        if site is not None:
            site_root = group_root / site

            if site_root.exists():
                search_roots.append(site_root)

        search_roots.append(group_root)

        matches: list[Path] = []

        for search_root in search_roots:
            patterns = (
                f"**/{subject_folder}/eeg/*.set",
                f"**/{subject_folder}/*.set",
            )

            for pattern in patterns:
                matches.extend(
                    search_root.glob(pattern)
                )

            if matches:
                break

        return sorted(
            {
                match.resolve()
                for match in matches
                if match.is_file()
            }
        )

    @classmethod
    def _find_eeg_file(
        cls,
        *,
        group_root: Path,
        subject_id: str,
        site: str | None,
    ) -> Path:
        """Find the unique EEGLAB SET file for a participant."""
        matches = cls._find_eeg_files(
            group_root=group_root,
            subject_id=subject_id,
            site=site,
        )

        if not matches:
            raise FileNotFoundError(
                "No EEGLAB .set file found for "
                f"sub-{subject_id} below {group_root}."
            )

        if len(matches) > 1:
            raise BrainLatDatasetError(
                f"Multiple .set files found for sub-{subject_id}: "
                f"{[str(path) for path in matches]}"
            )

        return matches[0]

    # -------------------------------------------------------------------------
    # MNE / EEGLAB loading
    # -------------------------------------------------------------------------

    @classmethod
    def _infer_auxiliary_channel_types(
        cls,
        raw: mne.io.Raw,
    ) -> None:
        """
        Mark EOG, ECG and EMG channels when their names identify them.

        High-density channels named A1-D32 remain EEG channels.
        """
        channel_types: dict[str, str] = {}

        for channel_name in raw.ch_names:
            normalized_name = channel_name.upper()

            for channel_type, patterns in (
                cls.AUXILIARY_CHANNEL_PATTERNS.items()
            ):
                if any(
                    pattern in normalized_name
                    for pattern in patterns
                ):
                    channel_types[channel_name] = channel_type
                    break

        if channel_types:
            raw.set_channel_types(
                channel_types,
                on_unit_change="ignore",
                verbose=False,
            )

    @staticmethod
    def _has_valid_channel_positions(
        raw: mne.io.Raw,
    ) -> bool:
        """Return whether at least one EEG channel has usable coordinates."""
        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            exclude=[],
        )

        for channel_index in eeg_picks:
            location = raw.info["chs"][channel_index]["loc"][:3]

            if not all(
                math.isfinite(float(value))
                for value in location
            ):
                continue

            if any(
                abs(float(value)) > 0
                for value in location
            ):
                return True

        return False

    @classmethod
    def _prepare_raw(
        cls,
        raw: mne.io.Raw,
    ) -> mne.io.Raw:
        """
        Prepare a BrainLat Raw object.

        The channel names and coordinates embedded in the EEGLAB file are
        preserved. No standard 10-20 montage is applied.
        """
        cls._infer_auxiliary_channel_types(raw)

        return raw

    @classmethod
    def _read_raw_eeglab(
        cls,
        eeg_path: Path,
        *,
        load_data: bool,
    ) -> mne.io.Raw:
        """Read one BrainLat EEGLAB recording."""
        errors: list[Exception] = []

        read_attempts = (
            {},
            {"uint16_codec": "latin1"},
            {"uint16_codec": "utf-8"},
        )

        for additional_arguments in read_attempts:
            try:
                raw = mne.io.read_raw_eeglab(
                    input_fname=eeg_path,
                    preload=load_data,
                    verbose=False,
                    **additional_arguments,
                )

                return cls._prepare_raw(raw)

            except Exception as error:
                errors.append(error)

        error_details = "\n".join(
            f"- {type(error).__name__}: {error}"
            for error in errors
        )

        raise BrainLatDatasetError(
            f"Unable to read EEGLAB file {eeg_path}.\n"
            f"MNE errors:\n{error_details}"
        )

    @classmethod
    def _make_raw_loader(
        cls,
        eeg_path: Path,
    ) -> RawLoader:
        """Create the lazy Raw loader used by EEGRecordedData."""

        def loader() -> mne.io.Raw:
            return cls._read_raw_eeglab(
                eeg_path=eeg_path,
                load_data=False,
            )

        return loader

    @classmethod
    def _build_recording(
        cls,
        *,
        participant: Participant,
        eeg_path: Path,
        load_data: bool,
    ) -> EEGRecordedData:
        """Build one EEGRecordedData object."""
        raw_preview = cls._read_raw_eeglab(
            eeg_path=eeg_path,
            load_data=load_data,
        )

        participant_metadata = participant.metadata
        participant_metadata["eeg_file"] = str(eeg_path)
        participant_metadata["n_channels"] = len(raw_preview.ch_names)
        participant_metadata["channel_names"] = list(
            raw_preview.ch_names
        )
        participant_metadata["has_channel_positions"] = (
            cls._has_valid_channel_positions(raw_preview)
        )

        enriched_participant = Participant(
            id=participant.id,
            gender=participant.gender,
            age=participant.age,
            group=participant.group,
            mmse=participant.mmse,
            moca=participant.moca,
            tag=participant.tag,
            metadata=participant_metadata,
        )

        return EEGRecordedData(
            raw=raw_preview,
            sampling_frequency=float(
                raw_preview.info["sfreq"]
            ),
            subject=enriched_participant,
            raw_loader=cls._make_raw_loader(eeg_path),
        )

    # -------------------------------------------------------------------------
    # Group and dataset construction
    # -------------------------------------------------------------------------

    @classmethod
    def _build_group(
        cls,
        *,
        dataset_root: Path,
        group: str,
        load_data: bool,
        skip_missing: bool,
    ) -> list[EEGRecordedData]:
        """Build all available recordings for one diagnostic group."""
        config = cls.GROUP_CONFIG[group]
        group_root = dataset_root / config["directory"]

        if not group_root.exists():
            raise FileNotFoundError(
                f"Missing BrainLat group directory: {group_root}"
            )

        records_path = group_root / config["records"]
        demographics_path = group_root / config["demographics"]
        cognition_path = group_root / config["cognition"]

        records = cls._index_rows_by_subject(
            cls._read_table(records_path),
            source_name=str(records_path),
        )

        demographics = cls._index_rows_by_subject(
            cls._read_table(demographics_path),
            source_name=str(demographics_path),
        )

        cognition = (
            cls._index_rows_by_subject(
                cls._read_table(cognition_path),
                source_name=str(cognition_path),
            )
            if cognition_path.exists()
            else {}
        )

        recordings: list[EEGRecordedData] = []

        for subject_id, record_row in records.items():
            if not cls._is_eeg_available(record_row):
                continue

            demographic_row = demographics.get(subject_id)

            if demographic_row is None:
                message = (
                    "Missing demographic metadata for "
                    f"sub-{subject_id}."
                )

                if skip_missing:
                    warnings.warn(
                        message,
                        RuntimeWarning,
                    )
                    continue

                raise BrainLatDatasetError(message)

            cognition_row = cognition.get(subject_id)

            try:
                participant = cls._extract_participant(
                    subject_id=subject_id,
                    expected_group=group,
                    record_row=record_row,
                    demographic_row=demographic_row,
                    cognition_row=cognition_row,
                )

                site = participant.get_metadata("site")

                eeg_path = cls._find_eeg_file(
                    group_root=group_root,
                    subject_id=subject_id,
                    site=site,
                )

                recording = cls._build_recording(
                    participant=participant,
                    eeg_path=eeg_path,
                    load_data=load_data,
                )

                recordings.append(recording)

            except (
                FileNotFoundError,
                BrainLatDatasetError,
            ) as error:
                if skip_missing:
                    warnings.warn(
                        f"Skipping sub-{subject_id}: {error}",
                        RuntimeWarning,
                    )
                    continue

                raise

        return recordings

    @staticmethod
    def _validate_unique_subjects(
        recordings: list[EEGRecordedData],
    ) -> None:
        """Ensure that each subject appears only once."""
        seen_subjects: set[tuple[str, str]] = set()

        for recording in recordings:
            subject_key = (
                recording.subject.group,
                recording.subject.id,
            )

            if subject_key in seen_subjects:
                raise BrainLatDatasetError(
                    "Duplicate BrainLat subject: "
                    f"group={subject_key[0]}, "
                    f"id={subject_key[1]}."
                )

            seen_subjects.add(subject_key)

    @classmethod
    def build(
        cls,
        data_file_path: str | Path,
        *,
        groups: tuple[str, ...] = ("AD", "CN"),
        load_data: bool = False,
        skip_missing: bool = False,
    ) -> list[EEGRecordedData]:
        """
        Build BrainLat Alzheimer and control EEG recordings.

        Parameters
        ----------
        data_file_path:
            Path to the BrainLat ``EEG data`` directory.

        groups:
            Diagnostic groups to load. Supported values are:

            - ``AD``;
            - ``CN``;
            - ``HC`` as an alias of ``CN``.

        load_data:
            If True, load EEG samples into memory immediately.

            If False, construct non-preloaded MNE Raw objects whenever the
            EEGLAB storage format permits it.

        skip_missing:
            If True, skip subjects with missing or unreadable files.

            If False, raise an exception at the first inconsistency.

        Returns
        -------
        list[EEGRecordedData]
            Recordings compatible with the existing EEG pipeline.
        """
        dataset_root = Path(
            data_file_path
        ).expanduser().resolve()

        if not dataset_root.exists():
            raise FileNotFoundError(
                f"BrainLat dataset directory does not exist: "
                f"{dataset_root}"
            )

        normalized_groups: list[str] = []

        for group in groups:
            normalized_group = cls._normalize_diagnosis(group)

            if normalized_group is None:
                raise ValueError(
                    f"Invalid BrainLat group: {group!r}."
                )

            if normalized_group not in cls.GROUP_CONFIG:
                raise ValueError(
                    f"Unsupported BrainLat group: {group!r}. "
                    "Supported groups are AD, CN and HC."
                )

            if normalized_group not in normalized_groups:
                normalized_groups.append(normalized_group)

        recordings: list[EEGRecordedData] = []

        for group in normalized_groups:
            recordings.extend(
                cls._build_group(
                    dataset_root=dataset_root,
                    group=group,
                    load_data=load_data,
                    skip_missing=skip_missing,
                )
            )

        cls._validate_unique_subjects(recordings)

        return recordings