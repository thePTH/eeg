from dataclasses import dataclass

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectGroupComparisonQuery(GroupComparisonQuery):
    """
    Statistical query for comparing a subject-level variable between two groups.

    Examples
    --------
    - Age: Healthy vs Alzheimer
    - MMSE: Healthy vs Alzheimer
    """

    variable: str

    @property
    def target_name(self) -> str:
        """Return the target subject-level variable."""
        return self.variable


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectCorrelationQuery(CorrelationQuery):
    """
    Statistical query for correlating two subject-level variables.

    Example
    -------
    - age vs mmse
    """

    x_variable: str
    y_variable: str

    @property
    def target_name(self) -> str:
        """Return the primary target variable."""
        return self.x_variable


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectFactorialQuery(FactorialQuery):
    """
    Statistical query for factorial analysis of a subject-level variable.

    Examples
    --------
    - subject_mmse ~ subject_health
    - subject_mmse ~ subject_health * subject_gender
    """

    variable: str

    @property
    def target_name(self) -> str:
        """Return the target subject-level variable."""
        return self.variable