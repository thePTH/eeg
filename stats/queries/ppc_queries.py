from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class PPCBandGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une bande PPC.

    Notes
    -----
    La granularité statistique est ici l'arête (paire de canaux).
    """
    band: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
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
    """
    Corrélation entre une bande PPC et une covariable sujet-level.
    """
    band: str
    covariate: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
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
    """
    Analyse factorielle sur une bande PPC.
    """
    band: str
    edge: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
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