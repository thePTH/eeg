from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True)
class PSDBandGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une bande PSD.
    """
    band: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "PSDBandGroupComparisonQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True)
class PSDBandCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une bande PSD et une covariable sujet-level.
    """
    band: str
    covariate: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "PSDBandCorrelationQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True)
class PSDBandFactorialQuery(FactorialQuery):
    """
    Analyse factorielle sur une bande PSD.
    """
    band: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.band

    def __post_init__(self):
        super().__post_init__()
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "PSDBandFactorialQuery requires "
                "scope='single_channel' or 'all_channels'"
            )