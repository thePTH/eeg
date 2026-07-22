from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from participants.genders import Gender
from participants.groups import Group
from utils.enum import EnumParser


class Participant:
    """
    Represent a participant from an EEG dataset.

    A participant can originate from different datasets. Cognitive scores are
    therefore optional and represented separately:

    - ``mmse`` for the Mini-Mental State Examination;
    - ``moca`` for the Montreal Cognitive Assessment.

    Additional dataset-specific information can be stored in ``metadata``.
    """

    def __init__(
        self,
        id: str,
        gender: Gender | str | None,
        age: int | float | None,
        group: Group | str,
        mmse: int | float | None = None,
        moca: int | float | None = None,
        tag: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ):
        """
        Initialize a participant.

        Parameters
        ----------
        id:
            Unique participant identifier, without the ``sub-`` prefix when
            possible.

        gender:
            Participant gender. It can be a Gender enum, a string accepted by
            EnumParser, or None when unavailable.

        age:
            Participant age. It can be None when unavailable.

        group:
            Diagnostic group. It can be a Group enum or a string accepted by
            EnumParser.

        mmse:
            Optional MMSE score.

        moca:
            Optional MoCA score.

        tag:
            Optional tag used to distinguish a participant or recording in a
            derived dataset.

        metadata:
            Optional dictionary containing dataset-specific information, such
            as acquisition site, education level, diagnosis details or other
            cognitive scores.
        """
        self._id = self._normalize_id(id)
        self._gender = self._parse_optional_gender(gender)
        self._age = self._normalize_optional_integer(age, field_name="age")
        self._group = EnumParser.parse(group, Group)

        self._mmse = self._normalize_optional_number(
            mmse,
            field_name="mmse",
        )
        self._moca = self._normalize_optional_number(
            moca,
            field_name="moca",
        )

        self._tag = self._normalize_optional_string(tag)
        self._metadata = dict(metadata) if metadata is not None else {}

    @staticmethod
    def _normalize_id(value: str) -> str:
        """Normalize and validate the participant identifier."""
        if value is None:
            raise ValueError("Participant id cannot be None.")

        participant_id = str(value).strip()

        if not participant_id:
            raise ValueError("Participant id cannot be empty.")

        if participant_id.lower().startswith("sub-"):
            participant_id = participant_id[4:]

        if not participant_id:
            raise ValueError(
                "Participant id cannot contain only the 'sub-' prefix."
            )

        return participant_id

    @staticmethod
    def _parse_optional_gender(
        gender: Gender | str | None,
    ) -> Gender | None:
        """Parse an optional participant gender."""
        if gender is None:
            return None

        if isinstance(gender, str):
            normalized_gender = gender.strip()

            if not normalized_gender:
                return None

            if normalized_gender.lower() in {
                "none",
                "nan",
                "na",
                "n/a",
                "unknown",
            }:
                return None

            gender = normalized_gender

        return EnumParser.parse(gender, Gender)

    @staticmethod
    def _normalize_optional_string(value: Any) -> str | None:
        """Normalize an optional string value."""
        if value is None:
            return None

        normalized_value = str(value).strip()

        if not normalized_value:
            return None

        if normalized_value.lower() in {
            "none",
            "nan",
            "na",
            "n/a",
        }:
            return None

        return normalized_value

    @staticmethod
    def _normalize_optional_number(
        value: int | float | str | None,
        *,
        field_name: str,
    ) -> int | float | None:
        """Normalize an optional numeric value."""
        if value is None:
            return None

        if isinstance(value, str):
            normalized_value = value.strip()

            if not normalized_value:
                return None

            if normalized_value.lower() in {
                "none",
                "nan",
                "na",
                "n/a",
            }:
                return None

            normalized_value = normalized_value.replace(",", ".")

            try:
                value = float(normalized_value)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value for {field_name}: {value!r}"
                ) from exc

        if isinstance(value, bool):
            raise TypeError(
                f"{field_name} must be a number or None, not a boolean."
            )

        if not isinstance(value, (int, float)):
            raise TypeError(
                f"{field_name} must be an int, float or None, "
                f"got {type(value).__name__}."
            )

        numeric_value = float(value)

        if numeric_value != numeric_value:
            return None

        if numeric_value.is_integer():
            return int(numeric_value)

        return numeric_value

    @classmethod
    def _normalize_optional_integer(
        cls,
        value: int | float | str | None,
        *,
        field_name: str,
    ) -> int | None:
        """Normalize an optional integer value."""
        normalized_value = cls._normalize_optional_number(
            value,
            field_name=field_name,
        )

        if normalized_value is None:
            return None

        numeric_value = float(normalized_value)

        if not numeric_value.is_integer():
            raise ValueError(
                f"{field_name} must be an integer, got {normalized_value!r}."
            )

        return int(numeric_value)

    @property
    def id(self) -> str:
        """Return the participant identifier."""
        return self._id

    @property
    def bids_id(self) -> str:
        """Return the participant identifier with the BIDS ``sub-`` prefix."""
        return f"sub-{self.id}"

    @property
    def gender(self) -> str | None:
        """Return the participant gender."""
        if self._gender is None:
            return None

        return self._gender.value

    @property
    def gender_enum(self) -> Gender | None:
        """Return the participant gender enum."""
        return self._gender

    @property
    def age(self) -> int | None:
        """Return the participant age."""
        return self._age

    @property
    def group(self) -> str:
        """Return the participant group."""
        return self._group.value

    @property
    def group_enum(self) -> Group:
        """Return the participant group enum."""
        return self._group

    @property
    def health_state(self) -> str:
        """Return the health state associated with the participant group."""
        return self._group.health_state.value

    @property
    def mmse(self) -> int | float | None:
        """Return the participant MMSE score."""
        return self._mmse

    @property
    def moca(self) -> int | float | None:
        """Return the participant MoCA score."""
        return self._moca

    @property
    def cognitive_score(self) -> int | float | None:
        """
        Return the principal available cognitive score.

        MMSE is returned first for backward compatibility. MoCA is returned
        when MMSE is unavailable.

        This property must not be used to compare raw scores across datasets
        without accounting for the fact that MMSE and MoCA are different
        clinical instruments.
        """
        if self.mmse is not None:
            return self.mmse

        return self.moca

    @property
    def cognitive_score_name(self) -> str | None:
        """Return the name of the principal available cognitive score."""
        if self.mmse is not None:
            return "mmse"

        if self.moca is not None:
            return "moca"

        return None

    @property
    def has_mmse(self) -> bool:
        """Return whether an MMSE score is available."""
        return self.mmse is not None

    @property
    def has_moca(self) -> bool:
        """Return whether a MoCA score is available."""
        return self.moca is not None

    @property
    def has_cognitive_score(self) -> bool:
        """Return whether at least one cognitive score is available."""
        return self.has_mmse or self.has_moca

    @property
    def tag(self) -> str | None:
        """Return the optional participant tag."""
        return self._tag

    @property
    def is_tagged(self) -> bool:
        """Return whether the participant has a tag."""
        return self.tag is not None

    @property
    def metadata(self) -> dict[str, Any]:
        """
        Return a copy of the participant metadata.

        A copy is returned to prevent external modifications of the internal
        participant state.
        """
        return deepcopy(self._metadata)

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """Return a metadata value."""
        return self._metadata.get(key, default)

    def with_tag(self, tag: str | None) -> Participant:
        """Return a copy of the participant with a different tag."""
        return Participant(
            id=self.id,
            gender=self.gender,
            age=self.age,
            group=self.group,
            mmse=self.mmse,
            moca=self.moca,
            tag=tag,
            metadata=self.metadata,
        )

    def copy(self) -> Participant:
        """Return a copy of the participant."""
        return Participant(
            id=self.id,
            gender=self.gender,
            age=self.age,
            group=self.group,
            mmse=self.mmse,
            moca=self.moca,
            tag=self.tag,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the participant to a serializable dictionary.

        Optional fields are always included. This produces a stable format for
        JSON exports and allows ParticipantFactory to load old and new
        participant representations.
        """
        return {
            "id": self.id,
            "gender": self.gender,
            "age": self.age,
            "group": self.group,
            "mmse": self.mmse,
            "moca": self.moca,
            "tag": self.tag,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        """Return a developer-friendly participant representation."""
        return (
            f"{type(self).__name__}("
            f"id={self.id!r}, "
            f"gender={self.gender!r}, "
            f"age={self.age!r}, "
            f"group={self.group!r}, "
            f"mmse={self.mmse!r}, "
            f"moca={self.moca!r}, "
            f"tag={self.tag!r}"
            f")"
        )


class ParticipantFactory:
    """Factory used to build Participant objects from dictionaries."""

    @staticmethod
    def build(dico: Mapping[str, Any]) -> Participant:
        """
        Build a participant from a dictionary.

        The factory supports both:

        - the previous format containing id, gender, age, group and mmse;
        - the new format containing optional mmse, moca, tag and metadata.
        """
        if not isinstance(dico, Mapping):
            raise TypeError(
                "ParticipantFactory.build expects a mapping, "
                f"got {type(dico).__name__}."
            )

        required_fields = {"id", "group"}
        missing_fields = required_fields.difference(dico.keys())

        if missing_fields:
            raise KeyError(
                "Missing required participant fields: "
                f"{sorted(missing_fields)}"
            )

        return Participant(
            id=dico["id"],
            gender=dico.get("gender"),
            age=dico.get("age"),
            group=dico["group"],
            mmse=dico.get("mmse"),
            moca=dico.get("moca"),
            tag=dico.get("tag"),
            metadata=dico.get("metadata", {}),
        )