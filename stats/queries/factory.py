from dataclasses import dataclass
from typing import Iterable


from typing import Final, Iterable, Literal, Optional

from .base import CorrelationQuery, FactorialQuery, GroupComparisonQuery
from .eeg_queries import (
    EEGFeatureCorrelationQuery,
    EEGFeatureFactorialQuery,
    EEGFeatureGroupComparisonQuery,
)
from .ppc_queries import (
    PPCBandCorrelationQuery,
    PPCBandFactorialQuery,
    PPCBandGroupComparisonQuery,
)
from .psd_queries import (
    PSDBandCorrelationQuery,
    PSDBandFactorialQuery,
    PSDBandGroupComparisonQuery,
)
from .specs import CorrectionSpec, PostHocSpec
from .subject_queries import (
    SubjectCorrelationQuery,
    SubjectFactorialQuery,
    SubjectGroupComparisonQuery,
)
from .types import Scope, TestKind


@dataclass(frozen=True)
class QueryFactoryConfig:
    """
    Configuration du QueryFactory.

    Cette configuration indique comment classifier un nom métier :
    - variable sujet
    - feature EEG scalaire
    - bande PSD
    - bande PPC

    Important
    ---------
    Les noms de bandes PSD et PPC peuvent se recouvrir (ex: theta, alpha, ...).
    L'ambiguïté est levée par le scope :
    - single_channel / all_channels -> PSD
    - single_edge / all_edges       -> PPC
    """
    subject_variables: frozenset[str]
    eeg_features: frozenset[str]
    psd_bands: frozenset[str]
    ppc_bands: frozenset[str]

    @classmethod
    def from_lists(
        cls,
        *,
        subject_variables: Iterable[str],
        eeg_features: Iterable[str],
        psd_bands: Iterable[str],
        ppc_bands: Iterable[str],
    ) -> "QueryFactoryConfig":
        return cls(
            subject_variables=frozenset(subject_variables),
            eeg_features=frozenset(eeg_features),
            psd_bands=frozenset(psd_bands),
            ppc_bands=frozenset(ppc_bands),
        )
    



class QueryFactory:
    """
    Factory métier pour construire automatiquement les bonnes query classes.
    """

    DEFAULT_GROUP_COLUMN: Final[str] = "subject_health"
    DEFAULT_NONPARAMETRIC_COMPARISON: Final[TestKind] = "wilcoxon_rank_sum"
    DEFAULT_PARAMETRIC_COMPARISON: Final[TestKind] = "t_test"
    DEFAULT_CORRELATION: Final[TestKind] = "spearman"

    def __init__(self, config: QueryFactoryConfig):
        self.config = config

    # ------------------------------------------------------------------------------
    # Public API - high level
    # ------------------------------------------------------------------------------

    def compare(
        self,
        *,
        target: str,
        group_a: str,
        group_b: str,
        scope: Scope | None = None,
        group_col: str | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
        parametric: bool = False,
    ) -> GroupComparisonQuery:
        test_kind: TestKind = (
            self.DEFAULT_PARAMETRIC_COMPARISON
            if parametric
            else self.DEFAULT_NONPARAMETRIC_COMPARISON
        )

        group_col = group_col or self.DEFAULT_GROUP_COLUMN
        name = self._normalize_name(target)

        scope = self._infer_scope_for_target(
            name=name,
            scope=scope,
            channel=channel,
            edge=edge,
        )

        is_subject = self._is_subject_variable(name)
        is_eeg = self._is_eeg_feature(name)
        is_psd = self._is_psd_band(name)
        is_ppc = self._is_ppc_band(name)

        if is_subject:
            self._validate_subject_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
            )
            return SubjectGroupComparisonQuery(
                variable=name,
                test_kind=test_kind,
                scope=scope,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                correction=correction,
            )

        if is_eeg:
            self._validate_channel_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="EEG feature",
            )
            return EEGFeatureGroupComparisonQuery(
                feature=name,
                test_kind=test_kind,
                scope=scope,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                channel=channel,
                correction=correction,
            )

        if is_psd and scope in {"single_channel", "all_channels"}:
            self._validate_channel_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="PSD band",
            )
            return PSDBandGroupComparisonQuery(
                band=name,
                test_kind=test_kind,
                scope=scope,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                channel=channel,
                correction=correction,
            )

        if is_ppc and scope in {"single_edge", "all_edges"}:
            self._validate_edge_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="PPC band",
            )
            return PPCBandGroupComparisonQuery(
                band=name,
                test_kind=test_kind,
                scope=scope,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                edge=edge,
                correction=correction,
            )

        raise ValueError(self._unknown_variable_message(name))

    def correlate(
        self,
        *,
        x: str,
        y: str,
        scope: Scope | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
    ) -> CorrelationQuery:
        test_kind: TestKind = self.DEFAULT_CORRELATION

        x_name = self._normalize_name(x)
        y_name = self._normalize_name(y)

        scope = self._infer_scope_for_correlation(
            x_name=x_name,
            y_name=y_name,
            scope=scope,
            channel=channel,
            edge=edge,
        )

        x_is_subject = self._is_subject_variable(x_name)
        y_is_subject = self._is_subject_variable(y_name)

        x_is_eeg = self._is_eeg_feature(x_name)
        y_is_eeg = self._is_eeg_feature(y_name)

        x_is_psd = self._is_psd_band(x_name)
        y_is_psd = self._is_psd_band(y_name)

        x_is_ppc = self._is_ppc_band(x_name)
        y_is_ppc = self._is_ppc_band(y_name)

        if scope == "subject":
            if x_is_subject and y_is_subject:
                self._validate_subject_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                )
                return SubjectCorrelationQuery(
                    x_variable=x_name,
                    y_variable=y_name,
                    test_kind=test_kind,
                    scope=scope,
                    correction=correction,
                )

            raise ValueError(
                "With scope='subject', both variables must be subject-level variables. "
                f"Known subject variables: {sorted(self.config.subject_variables)}."
            )

        if scope in {"single_channel", "all_channels"}:
            if x_is_eeg and y_is_subject:
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="EEG feature",
                )
                return EEGFeatureCorrelationQuery(
                    feature=x_name,
                    covariate=y_name,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                    correction=correction,
                )

            if x_is_subject and y_is_eeg:
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="EEG feature",
                )
                return EEGFeatureCorrelationQuery(
                    feature=y_name,
                    covariate=x_name,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                    correction=correction,
                )

            if x_is_psd and y_is_subject:
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="PSD band",
                )
                return PSDBandCorrelationQuery(
                    band=x_name,
                    covariate=y_name,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                    correction=correction,
                )

            if x_is_subject and y_is_psd:
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="PSD band",
                )
                return PSDBandCorrelationQuery(
                    band=y_name,
                    covariate=x_name,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                    correction=correction,
                )

            raise ValueError(
                "With scope='single_channel' or 'all_channels', supported correlations are "
                "only EEG-subject and PSD-subject."
            )

        if scope in {"single_edge", "all_edges"}:
            if x_is_ppc and y_is_subject:
                self._validate_edge_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="PPC band",
                )
                return PPCBandCorrelationQuery(
                    band=x_name,
                    covariate=y_name,
                    test_kind=test_kind,
                    scope=scope,
                    edge=edge,
                    correction=correction,
                )

            if x_is_subject and y_is_ppc:
                self._validate_edge_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=f"{x_name} vs {y_name}",
                    family="PPC band",
                )
                return PPCBandCorrelationQuery(
                    band=y_name,
                    covariate=x_name,
                    test_kind=test_kind,
                    scope=scope,
                    edge=edge,
                    correction=correction,
                )

            raise ValueError(
                "With scope='single_edge' or 'all_edges', supported correlations are "
                "only PPC-subject."
            )

        raise ValueError(f"Unsupported scope: {scope}")

    def factorial(
        self,
        *,
        target: str,
        factors: Iterable[str],
        scope: Scope | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
        posthoc: PostHocSpec | None = None,
    ) -> FactorialQuery:
        name = self._normalize_name(target)
        normalized_factors = tuple(self._normalize_name(f) for f in factors)

        if len(normalized_factors) == 1:
            test_kind: TestKind = "one_way_anova"
        elif len(normalized_factors) == 2:
            test_kind = "two_way_anova"
        else:
            raise ValueError(
                "factorial(...) supports only one-way or two-way designs, "
                f"but received {len(normalized_factors)} factors."
            )

        self._validate_factors_are_subject_level(normalized_factors)

        scope = self._infer_scope_for_target(
            name=name,
            scope=scope,
            channel=channel,
            edge=edge,
        )

        is_subject = self._is_subject_variable(name)
        is_eeg = self._is_eeg_feature(name)
        is_psd = self._is_psd_band(name)
        is_ppc = self._is_ppc_band(name)

        if is_subject:
            self._validate_subject_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
            )
            return SubjectFactorialQuery(
                variable=name,
                factors=normalized_factors,
                test_kind=test_kind,
                scope=scope,
                correction=correction,
                posthoc=posthoc,
            )

        if is_eeg:
            self._validate_channel_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="EEG feature",
            )
            return EEGFeatureFactorialQuery(
                feature=name,
                factors=normalized_factors,
                test_kind=test_kind,
                scope=scope,
                channel=channel,
                correction=correction,
                posthoc=posthoc,
            )

        if is_psd and scope in {"single_channel", "all_channels"}:
            self._validate_channel_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="PSD band",
            )
            return PSDBandFactorialQuery(
                band=name,
                factors=normalized_factors,
                test_kind=test_kind,
                scope=scope,
                channel=channel,
                correction=correction,
                posthoc=posthoc,
            )

        if is_ppc and scope in {"single_edge", "all_edges"}:
            self._validate_edge_scope(
                scope=scope,
                channel=channel,
                edge=edge,
                name=name,
                family="PPC band",
            )
            return PPCBandFactorialQuery(
                band=name,
                factors=normalized_factors,
                test_kind=test_kind,
                scope=scope,
                edge=edge,
                correction=correction,
                posthoc=posthoc,
            )

        raise ValueError(self._unknown_variable_message(name))

    # ------------------------------------------------------------------------------
    # Public API - explicit aliases
    # ------------------------------------------------------------------------------

    def one_way_anova(
        self,
        *,
        target: str,
        factor: str,
        scope: Scope | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
        posthoc: PostHocSpec | None = None,
    ) -> FactorialQuery:
        return self.factorial(
            target=target,
            factors=(factor,),
            scope=scope,
            channel=channel,
            edge=edge,
            correction=correction,
            posthoc=posthoc,
        )

    def two_way_anova(
        self,
        *,
        target: str,
        factor_a: str,
        factor_b: str,
        scope: Scope | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
        posthoc: PostHocSpec | None = None,
    ) -> FactorialQuery:
        return self.factorial(
            target=target,
            factors=(factor_a, factor_b),
            scope=scope,
            channel=channel,
            edge=edge,
            correction=correction,
            posthoc=posthoc,
        )

    # ------------------------------------------------------------------------------
    # Backward-compatible explicit constructors
    # ------------------------------------------------------------------------------

    def compare_with_test(
        self,
        *,
        target: str,
        group_a: str,
        group_b: str,
        test_kind: Literal["t_test", "wilcoxon_rank_sum"],
        scope: Scope | None = None,
        group_col: str | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
    ) -> GroupComparisonQuery:
        return self.compare(
            target=target,
            group_a=group_a,
            group_b=group_b,
            scope=scope,
            group_col=group_col,
            channel=channel,
            edge=edge,
            correction=correction,
            parametric=(test_kind == "t_test"),
        )

    def correlate_with_test(
        self,
        *,
        x: str,
        y: str,
        test_kind: Literal["spearman"],
        scope: Scope | None = None,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
        correction: CorrectionSpec | None = None,
    ) -> CorrelationQuery:
        if test_kind != "spearman":
            raise ValueError("Only 'spearman' is currently supported for correlations")
        return self.correlate(
            x=x,
            y=y,
            scope=scope,
            channel=channel,
            edge=edge,
            correction=correction,
        )

    # ------------------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "QueryFactory":
        config = QueryFactoryConfig.from_lists(
            subject_variables={
                "age",
                "mmse",
                "education_years",
                "subject_mmse",
                "subject_age",
                "subject_health",
                "subject_group",
                "subject_gender",
            },
            eeg_features={
                "variance",
                "skewness",
                "kurtosis",
                "peak_amplitude",
                "shape_factor",
                "impulse_factor",
                "crest_factor",
                "clearance_factor",
                "willison_amplitude",
                "zero_crossing_rate",
                "sample_entropy",
                "approximate_entropy",
                "permutation_entropy",
                "state_space_correlation_entropy",
                "correlation_dimension",
                "higuchi_fd",
                "katz_fd",
                "lyapunov_exponent",
                "hurst_exponent",
                "lz_complexity",
                "hjorth_activity",
                "hjorth_mobility",
                "hjorth_complexity",
                "alpha_dominant_frequency",
                "gamma_dominant_frequency",
                "spectral_rolloff",
                "spectral_centroid",
                "spectral_spread",
                "spectral_flux",
                "spectral_skewness",
                "spectral_kurtosis",
                "theta_beta_ratio",
                "theta_alpha_ratio",
                "gamma_alpha_ratio",
                "delta_alpha_ratio",
                "spectral_power_ratio",
                "wavelet_energy_approximation",
                "wavelet_energy_detail",
                "relative_wavelet_energy",
                "wavelet_packet_energy_approximation",
                "wavelet_packet_energy_detail",
                "relative_wavelet_packet_energy",
            },
            psd_bands={
                "delta",
                "theta",
                "alpha",
                "beta",
                "gamma",
                "full",
            },
            ppc_bands={
                "delta",
                "theta",
                "alpha",
                "beta",
                "gamma",
                "full",
            },
        )
        return cls(config=config)

    @classmethod
    def from_dataset_metadata(
        cls,
        *,
        subject_variables: Iterable[str],
        eeg_features: Iterable[str],
        psd_bands: Iterable[str],
        ppc_bands: Iterable[str],
    ) -> "QueryFactory":
        return cls(
            config=QueryFactoryConfig.from_lists(
                subject_variables=subject_variables,
                eeg_features=eeg_features,
                psd_bands=psd_bands,
                ppc_bands=ppc_bands,
            )
        )

    # ------------------------------------------------------------------------------
    # Internal helpers - inference
    # ------------------------------------------------------------------------------

    def _infer_scope_for_target(
        self,
        *,
        name: str,
        scope: Scope | None,
        channel: Optional[str],
        edge: Optional[str],
    ) -> Scope:
        if scope is not None:
            return scope

        if channel is not None and edge is not None:
            raise ValueError(
                "Cannot infer scope automatically when both channel and edge are provided."
            )

        is_subject = self._is_subject_variable(name)
        is_eeg = self._is_eeg_feature(name)
        is_psd = self._is_psd_band(name)
        is_ppc = self._is_ppc_band(name)

        if is_subject:
            if channel is not None or edge is not None:
                raise ValueError(
                    f"'{name}' is a subject-level variable, so channel and edge must be None."
                )
            return "subject"

        if is_psd and is_ppc:
            if channel is not None:
                return "single_channel"
            if edge is not None:
                return "single_edge"
            raise ValueError(
                f"Variable '{name}' is ambiguous because it is registered as both a PSD band "
                "and a PPC band. Please specify `scope` explicitly, or provide `channel` "
                "for channel-based analysis, or `edge` for edge-based analysis."
            )

        if is_eeg or is_psd:
            if edge is not None:
                raise ValueError(f"'{name}' is channel-based, so edge must be None.")
            return "single_channel" if channel is not None else "all_channels"

        if is_ppc:
            if channel is not None:
                raise ValueError(f"'{name}' is edge-based, so channel must be None.")
            return "single_edge" if edge is not None else "all_edges"

        raise ValueError(self._unknown_variable_message(name))

    def _infer_scope_for_correlation(
        self,
        *,
        x_name: str,
        y_name: str,
        scope: Scope | None,
        channel: Optional[str],
        edge: Optional[str],
    ) -> Scope:
        if scope is not None:
            return scope

        if channel is not None and edge is not None:
            raise ValueError(
                "Cannot infer scope automatically when both channel and edge are provided."
            )

        x_is_subject = self._is_subject_variable(x_name)
        y_is_subject = self._is_subject_variable(y_name)

        x_is_eeg = self._is_eeg_feature(x_name)
        y_is_eeg = self._is_eeg_feature(y_name)

        x_is_psd = self._is_psd_band(x_name)
        y_is_psd = self._is_psd_band(y_name)

        x_is_ppc = self._is_ppc_band(x_name)
        y_is_ppc = self._is_ppc_band(y_name)

        if x_is_subject and y_is_subject:
            if channel is not None or edge is not None:
                raise ValueError(
                    "Subject-vs-subject correlation cannot receive channel or edge."
                )
            return "subject"

        if (x_is_eeg and y_is_subject) or (x_is_subject and y_is_eeg):
            if edge is not None:
                raise ValueError(
                    "EEG-subject correlation is channel-based, so edge must be None."
                )
            return "single_channel" if channel is not None else "all_channels"

        if (x_is_psd and y_is_subject) or (x_is_subject and y_is_psd):
            if edge is not None:
                raise ValueError(
                    "PSD-subject correlation is channel-based, so edge must be None."
                )
            return "single_channel" if channel is not None else "all_channels"

        if (x_is_ppc and y_is_subject) or (x_is_subject and y_is_ppc):
            if channel is not None:
                raise ValueError(
                    "PPC-subject correlation is edge-based, so channel must be None."
                )
            return "single_edge" if edge is not None else "all_edges"

        raise ValueError(
            "Unsupported correlation. Supported cases are only: "
            "subject-subject, EEG-subject, PSD-subject, PPC-subject."
        )

    # ------------------------------------------------------------------------------
    # Internal helpers - metadata checks
    # ------------------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        normalized = name.strip()
        if not normalized:
            raise ValueError("Variable name cannot be empty.")
        return normalized

    def _is_subject_variable(self, name: str) -> bool:
        return name in self.config.subject_variables

    def _is_eeg_feature(self, name: str) -> bool:
        return name in self.config.eeg_features

    def _is_psd_band(self, name: str) -> bool:
        return name in self.config.psd_bands

    def _is_ppc_band(self, name: str) -> bool:
        return name in self.config.ppc_bands

    def _validate_factors_are_subject_level(self, factors: tuple[str, ...]) -> None:
        unknown = [factor for factor in factors if factor not in self.config.subject_variables]
        if unknown:
            raise ValueError(
                "All factors must be subject-level variables. "
                f"Unknown factors: {unknown}. "
                f"Known subject variables: {sorted(self.config.subject_variables)}."
            )

    # ------------------------------------------------------------------------------
    # Internal helpers - scope validation
    # ------------------------------------------------------------------------------

    def _validate_subject_scope(
        self,
        *,
        scope: Scope,
        channel: Optional[str],
        edge: Optional[str],
        name: str,
    ) -> None:
        if scope != "subject":
            raise ValueError(
                f"'{name}' is a subject-level variable and therefore requires "
                "scope='subject'."
            )
        if channel is not None:
            raise ValueError(
                f"'{name}' is a subject-level variable, so channel must be None."
            )
        if edge is not None:
            raise ValueError(
                f"'{name}' is a subject-level variable, so edge must be None."
            )

    def _validate_channel_scope(
        self,
        *,
        scope: Scope,
        channel: Optional[str],
        edge: Optional[str],
        name: str,
        family: str,
    ) -> None:
        if scope not in {"single_channel", "all_channels"}:
            raise ValueError(
                f"'{name}' is a {family} and therefore requires "
                "scope='single_channel' or scope='all_channels'."
            )

        if edge is not None:
            raise ValueError(f"'{name}' is a {family}, so edge must be None.")

        if scope == "single_channel" and channel is None:
            raise ValueError(f"'{name}' requires a channel when scope='single_channel'.")

        if scope == "all_channels" and channel is not None:
            raise ValueError(
                f"'{name}' received channel='{channel}', but channel is only allowed "
                "when scope='single_channel'."
            )

    def _validate_edge_scope(
        self,
        *,
        scope: Scope,
        channel: Optional[str],
        edge: Optional[str],
        name: str,
        family: str,
    ) -> None:
        if scope not in {"single_edge", "all_edges"}:
            raise ValueError(
                f"'{name}' is a {family} and therefore requires "
                "scope='single_edge' or scope='all_edges'."
            )

        if channel is not None:
            raise ValueError(f"'{name}' is a {family}, so channel must be None.")

        if scope == "single_edge" and edge is None:
            raise ValueError(f"'{name}' requires an edge when scope='single_edge'.")

        if scope == "all_edges" and edge is not None:
            raise ValueError(
                f"'{name}' received edge='{edge}', but edge is only allowed "
                "when scope='single_edge'."
            )

    def _unknown_variable_message(self, name: str) -> str:
        return (
            f"Unknown variable '{name}'. "
            f"Known subject variables: {sorted(self.config.subject_variables)}. "
            f"Known EEG features: {sorted(self.config.eeg_features)}. "
            f"Known PSD bands: {sorted(self.config.psd_bands)}. "
            f"Known PPC bands: {sorted(self.config.ppc_bands)}."
        )