from abc import ABC
from dataclasses import dataclass

from .specs import CorrectionSpec, PostHocSpec
from .types import Scope, TestKind


@dataclass(frozen=True, kw_only=True)
class StatisticalQuery(ABC):
    """
    Classe racine de toutes les requêtes statistiques.

    Une query est une description métier :
    - quelle cible est testée
    - avec quelle portée (scope)
    - avec quel test
    - avec quelle correction éventuelle
    """
    test_kind: TestKind
    scope: Scope
    correction: CorrectionSpec | None = None

    @property
    def target_name(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, kw_only=True)
class GroupComparisonQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour comparer deux groupes.
    """
    group_col: str
    group_a: str
    group_b: str


@dataclass(frozen=True, kw_only=True)
class CorrelationQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour corrélation entre deux variables.
    """
    pass


@dataclass(frozen=True, kw_only=True)
class FactorialQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour design factoriel.

    Convention
    ----------
    - one_way_anova <=> len(factors) == 1
    - two_way_anova <=> len(factors) == 2
    """
    factors: tuple[str, ...]
    posthoc: PostHocSpec | None = None

    def __post_init__(self):
        if self.test_kind == "one_way_anova" and len(self.factors) != 1:
            raise ValueError("one_way_anova requires exactly one factor")
        if self.test_kind == "two_way_anova" and len(self.factors) != 2:
            raise ValueError("two_way_anova requires exactly two factors")
        if self.test_kind not in {"one_way_anova", "two_way_anova"}:
            raise ValueError(
                "FactorialQuery requires test_kind='one_way_anova' or 'two_way_anova'"
            )

    @property
    def factor_names(self) -> tuple[str, ...]:
        return self.factors