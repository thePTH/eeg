from dataclasses import dataclass
from typing import Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery


@dataclass(frozen=True, kw_only=True, repr=False)
class PSDBandGroupComparisonQuery(GroupComparisonQuery):
    """Statistical query for comparing a PSD band between two groups."""

    band: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PSD band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")

        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")

        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "PSDBandGroupComparisonQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class PSDBandCorrelationQuery(CorrelationQuery):
    """Statistical query for correlating a PSD band with a subject-level covariate."""

    band: str
    covariate: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PSD band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")

        if self.scope == "all_channels" and self.channel is not None:
            raise ValueError("channel must be None when scope='all_channels'")

        if self.scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                "PSDBandCorrelationQuery requires "
                "scope='single_channel' or 'all_channels'"
            )


@dataclass(frozen=True, kw_only=True, repr=False)
class PSDBandFactorialQuery(FactorialQuery):
    """Statistical query for factorial analysis of a PSD band."""

    band: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        """Return the target PSD band."""
        return self.band

    def __post_init__(self):
        """Validate the consistency between the selected scope and channel."""
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