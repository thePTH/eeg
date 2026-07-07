from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class PPCBandGroupComparisonQuery(GroupComparisonQuery):
    """
    Statistical query for comparing a PPC band between two groups.

    Notes
    -----
    The statistical granularity is the edge, that is, a pair of channels.
    """

    band: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PPC band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and edge."""
        if self.scope == "single_edge" and self.edge is None:
            raise ValueError("edge must be provided when scope='single_edge'")

        if self.scope == "all_edges" and self.edge is not None:
            raise ValueError("edge must be None when scope='all_edges'")

        if self.scope not in {"single_edge", "all_edges"}:
            raise ValueError(
                "PPCBandGroupComparisonQuery requires "
                "scope='single_edge' or 'all_edges'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class PPCBandCorrelationQuery(CorrelationQuery):
    """Statistical query for correlating a PPC band with a subject-level covariate."""

    band: str
    covariate: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PPC band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and edge."""
        if self.scope == "single_edge" and self.edge is None:
            raise ValueError("edge must be provided when scope='single_edge'")

        if self.scope == "all_edges" and self.edge is not None:
            raise ValueError("edge must be None when scope='all_edges'")

        if self.scope not in {"single_edge", "all_edges"}:
            raise ValueError(
                "PPCBandCorrelationQuery requires "
                "scope='single_edge' or 'all_edges'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class PPCBandFactorialQuery(FactorialQuery):
    """Statistical query for factorial analysis of a PPC band."""

    band: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PPC band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and edge."""
        super().__post_init__()

        if self.scope == "single_edge" and self.edge is None:
            raise ValueError("edge must be provided when scope='single_edge'")

        if self.scope == "all_edges" and self.edge is not None:
            raise ValueError("edge must be None when scope='all_edges'")

        if self.scope not in {"single_edge", "all_edges"}:
            raise ValueError(
                "PPCBandFactorialQuery requires "
                "scope='single_edge' or 'all_edges'"
            )