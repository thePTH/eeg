from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une feature EEG scalaire.
    """
    feature: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.feature

    def __post_init__(self):
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureGroupComparisonQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une feature EEG et une covariable sujet-level.
    """
    feature: str
    covariate: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.feature

    def __post_init__(self):
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureCorrelationQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureFactorialQuery(FactorialQuery):
    """
    Analyse factorielle sur une feature EEG.
    """
    feature: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.feature

    def __post_init__(self):
        super().__post_init__()
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")
        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureFactorialQuery requires "
                "scope='single_channel' or 'all_channels'"
            )