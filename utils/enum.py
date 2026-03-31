from enum import Enum

class EnumParser:
    @staticmethod
    def parse(value, enum_cls: type[Enum]):
        """
        Convert value to enum_cls if possible.

        Accepts:
            - enum instance
            - enum name (e.g. "AD")
            - enum value (e.g. "A")

        Raises:
            ValueError if conversion fails.
        """

        if isinstance(value, enum_cls):
            return value

        if isinstance(value, str):
            try:
                return enum_cls[value]
            except KeyError:
                try:
                    return enum_cls(value)
                except ValueError:
                    pass

        raise ValueError(f"{value} is not a valid {enum_cls.__name__}")