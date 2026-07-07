from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureGroupComparisonQuery(GroupComparisonQuery):
    """
    Statistical query for comparing an EEG feature between two groups.

    The analysis can target either a single EEG channel or all available
    channels, depending on the selected scope.
    """

    feature: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target EEG feature."""
        return self.feature

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError(
                "channel must be provided when scope='single_channel'"
            )

        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError(
                "channel must be None when scope='all_channels'"
            )

        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureGroupComparisonQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureCorrelationQuery(CorrelationQuery):
    """
    Statistical query for correlating an EEG feature with a subject-level
    covariate.

    The analysis can target either a single EEG channel or all available
    channels.
    """

    feature: str
    covariate: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target EEG feature."""
        return self.feature

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError(
                "channel must be provided when scope='single_channel'"
            )

        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError(
                "channel must be None when scope='all_channels'"
            )

        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureCorrelationQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class EEGFeatureFactorialQuery(FactorialQuery):
    """
    Statistical query for factorial analysis of an EEG feature.

    The analysis can target either a single EEG channel or all available
    channels.
    """

    feature: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target EEG feature."""
        return self.feature

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
        super().__post_init__()

        if self.scope == "single_channel" and self.channel is None:
            raise ValueError(
                "channel must be provided when scope='single_channel'"
            )

        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError(
                "channel must be None when scope='all_channels'"
            )

        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "EEGFeatureFactorialQuery requires "
                "scope='single_channel' or 'all_channels'"
            )