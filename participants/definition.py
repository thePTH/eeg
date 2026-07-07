from typing import Union

from participants.genders import Gender
from participants.groups import Group
from utils.enum import EnumParser


class Participant:
    """Represent a participant from the EEG dataset."""

    def __init__(
        self,
        id: str,
        gender: Union[Gender, str],
        age: int,
        group: Union[Group, str],
        mmse: int,
        tag: str = None,
    ):
        self._gender = EnumParser.parse(gender, Gender)
        self._group = EnumParser.parse(group, Group)

        self._id = id
        self._age = age
        self._mmse = mmse
        self._tag = tag

    @property
    def id(self):
        """Return the participant identifier."""
        return self._id

    @property
    def gender(self) -> str:
        """Return the participant gender."""
        return self._gender.value

    @property
    def age(self):
        """Return the participant age."""
        return self._age

    @property
    def group(self) -> str:
        """Return the participant group."""
        return self._group.value

    @property
    def health_state(self) -> str:
        """Return the health state associated with the participant group."""
        return self._group.health_state.value

    @property
    def mmse(self):
        """Return the participant MMSE score."""
        return self._mmse

    @property
    def tag(self):
        """Return the optional participant tag."""
        return self._tag

    @property
    def is_tagged(self):
        """Return whether the participant has a tag."""
        return bool(self.tag)

    def to_dict(self):
        """Convert the participant to a dictionary."""
        if self.is_tagged:
            return {
                "id": self.id,
                "gender": self.gender,
                "age": self.age,
                "group": self.group,
                "mmse": self.mmse,
                "tag": self.tag,
            }

        return {
            "id": self.id,
            "gender": self.gender,
            "age": self.age,
            "group": self.group,
            "mmse": self.mmse,
        }


class ParticipantFactory:
    """Factory used to build Participant objects from dictionaries."""

    @staticmethod
    def build(dico: dict):
        """Build a participant from a dictionary."""
        tag = dico["tag"] if "tag" in dico.keys() else None

        return Participant(
            id=dico["id"],
            gender=dico["gender"],
            age=dico["age"],
            group=dico["group"],
            mmse=dico["mmse"],
            tag=tag,
        )