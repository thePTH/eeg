from dataclasses import dataclass

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une variable sujet-level.

    Exemples
    --------
    - âge Healthy vs Alzheimer
    - MMSE Healthy vs Alzheimer
    """
    variable: str

    @property
    def target_name(self) -> str:
        return self.variable


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre deux variables sujet-level.

    Exemple
    -------
    - age vs mmse
    """
    x_variable: str
    y_variable: str

    @property
    def target_name(self) -> str:
        return self.x_variable


@dataclass(frozen=True, kw_only=True, repr=False)
class SubjectFactorialQuery(FactorialQuery):
    """
    Analyse factorielle sur variable sujet-level.

    Exemples
    --------
    - subject_mmse ~ subject_health
    - subject_mmse ~ subject_health * subject_gender
    """
    variable: str

    @property
    def target_name(self) -> str:
        return self.variable