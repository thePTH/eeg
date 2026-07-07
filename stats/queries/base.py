from abc import ABC
from dataclasses import dataclass, fields

from .specs import CorrectionSpec, PostHocSpec
from .types import Scope, TestKind


@dataclass(frozen=True, kw_only=True, repr=False)
class StatisticalQuery(ABC):
    """
    Base class for all statistical queries.

    A statistical query describes:
    - the target variable;
    - the analysis scope;
    - the statistical test to perform;
    - an optional multiple-comparison correction.
    """

    test_kind: TestKind
    scope: Scope
    correction: CorrectionSpec | None = None

    @property
    def target_name(self) -> str:
        """Return the name of the target variable."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return a compact and readable representation suitable for debugging,
        logging, and interactive notebooks.

        Example
        -------
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
        """Return the string representation of the query."""
        return self.__repr__()


@dataclass(frozen=True, kw_only=True, repr=False)
class GroupComparisonQuery(StatisticalQuery, ABC):
    """Abstract query for comparing two groups."""

    group_col: str
    group_a: str
    group_b: str


@dataclass(frozen=True, kw_only=True, repr=False)
class CorrelationQuery(StatisticalQuery, ABC):
    """Abstract query describing a correlation analysis."""

    pass


@dataclass(frozen=True, kw_only=True, repr=False)
class FactorialQuery(StatisticalQuery, ABC):
    """
    Abstract query describing a factorial design.

    Convention
    ----------
    - one_way_anova  -> exactly one factor
    - two_way_anova  -> exactly two factors
    """

    factors: tuple[str, ...]
    posthoc: PostHocSpec | None = None

    def __post_init__(self):
        """Validate the consistency between the selected test and the number of factors."""
        if self.test_kind == "one_way_anova" and len(self.factors) != 1:
            raise ValueError("one_way_anova requires exactly one factor")

        if self.test_kind == "two_way_anova" and len(self.factors) != 2:
            raise ValueError("two_way_anova requires exactly two factors")

        if self.test_kind not in {"one_way_anova", "two_way_anova"}:
            raise ValueError(
                "FactorialQuery requires "
                "test_kind='one_way_anova' or 'two_way_anova'"
            )

    @property
    def factor_names(self) -> tuple[str, ...]:
        """Return the names of the experimental factors."""
        return self.factors