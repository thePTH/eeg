from abc import ABC
from dataclasses import dataclass, fields

from .specs import CorrectionSpec, PostHocSpec
from .types import Scope, TestKind


@dataclass(frozen=True, kw_only=True, repr=False)
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

    def __repr__(self) -> str:
        """
        Représentation compacte et lisible pour debug / logs / notebook.

        Exemple
        --------
        SubjectGroupComparisonQuery(
            target=mmse,
            test=wilcoxon_rank_sum,
            scope=subject,
            group_col=subject_health,
            group_a=Healthy,
            group_b=Alzheimer
        )
        """

        class_name = self.__class__.__name__

        core = [
            f"target={self.target_name}",
            f"test={self.test_kind}",
            f"scope={self.scope}",
        ]

        extras = []

        for f in fields(self):
            name = f.name

            if name in {"test_kind", "scope", "correction"}:
                continue

            value = getattr(self, name)

            if value is not None:
                extras.append(f"{name}={value}")

        if self.correction is not None:
            extras.append(f"correction={self.correction.method}")

        args = ", ".join(core + extras)

        return f"{class_name}({args})"
    
    def __str__(self):
        return self.__repr__()


@dataclass(frozen=True, kw_only=True, repr=False)
class GroupComparisonQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour comparer deux groupes.
    """
    group_col: str
    group_a: str
    group_b: str


@dataclass(frozen=True, kw_only=True, repr=False)
class CorrelationQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour corrélation entre deux variables.
    """
    pass


@dataclass(frozen=True, kw_only=True, repr=False)
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