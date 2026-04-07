from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Final, Iterable, Literal, Optional


# ======================================================================================
#                                      TYPES
# ======================================================================================

Scope = Literal[
    "subject",
    "single_channel",
    "all_channels",
    "single_edge",
    "all_edges",
]

TestKind = Literal[
    "t_test",
    "wilcoxon_rank_sum",
    "spearman",
]


# ======================================================================================
#                                      QUERIES
# ======================================================================================

@dataclass(frozen=True)
class StatisticalQuery(ABC):
    """
    Classe abstraite racine pour toutes les requêtes statistiques.

    Une query représente une intention métier indépendante de l'implémentation
    statistique sous-jacente.

    Elle répond à des questions du type :
    - Quelle variable / feature veut-on tester ?
    - Avec quelle portée (niveau sujet, canal unique, tous les canaux,
      arête unique, toutes les arêtes) ?
    - Avec quel test statistique ?
    """
    test_kind: TestKind
    scope: Scope

    @property
    def target_name(self) -> str:
        """
        Nom métier principal de la cible testée.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GroupComparisonQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour comparer deux groupes.
    """
    group_col: str
    group_a: str
    group_b: str


@dataclass(frozen=True)
class CorrelationQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour corrélation entre deux variables.
    """
    pass


# ======================================================================================
#                           SUBJECT-LEVEL QUERIES
# ======================================================================================

@dataclass(frozen=True)
class SubjectGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une variable sujet-level.

    Exemples
    --------
    - âge Healthy vs Alzheimer
    - MMSE Healthy vs Alzheimer
    - années d'éducation Healthy vs Alzheimer
    """
    variable: str

    @property
    def target_name(self) -> str:
        return self.variable


@dataclass(frozen=True)
class SubjectCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre deux variables sujet-level.

    Exemple
    -------
    - age vs mmse
    """
    x_variable: str
    y_variable: str

    @property
    def target_name(self) -> str:
        return self.x_variable


# ======================================================================================
#                           EEG FEATURE QUERIES
# ======================================================================================

@dataclass(frozen=True)
class EEGFeatureGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une feature EEG scalaire.

    Exemples
    --------
    - theta_beta_ratio Healthy vs Alzheimer sur tous les canaux
    - gamma_alpha_ratio Healthy vs Prodromal sur un canal donné
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


@dataclass(frozen=True)
class EEGFeatureCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une feature EEG et une covariable sujet-level.

    Exemples
    --------
    - theta_alpha_ratio vs subject_mmse sur tous les canaux
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


# ======================================================================================
#                                 PSD QUERIES
# ======================================================================================

@dataclass(frozen=True)
class PSDBandGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une bande PSD.

    Exemples
    --------
    - theta Healthy vs Alzheimer sur tous les canaux
    - gamma Healthy vs Prodromal sur Pz
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


@dataclass(frozen=True)
class PSDBandCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une bande PSD et une covariable sujet-level.

    Exemples
    --------
    - theta vs subject_mmse sur tous les canaux
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


# ======================================================================================
#                                 PPC QUERIES
# ======================================================================================

@dataclass(frozen=True)
class PPCBandGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une bande PPC.

    Notes
    -----
    La granularité statistique est ici l'arête (paire de canaux), et non la fréquence.

    Exemples
    --------
    - theta Healthy vs Alzheimer sur toutes les arêtes
    - gamma Healthy vs Prodromal sur l'arête Fp1__Fp2
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


@dataclass(frozen=True)
class PPCBandCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une bande PPC et une covariable sujet-level.

    Exemples
    --------
    - theta vs subject_mmse sur toutes les arêtes
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


# ======================================================================================
#                               QUERY FACTORY CONFIG
# ======================================================================================

@dataclass(frozen=True)
class QueryFactoryConfig:
    """
    Configuration du QueryFactory.

    Cette configuration sert à indiquer au factory comment distinguer :
    - les variables sujet-level
    - les features EEG scalaires
    - les bandes PSD
    - les bandes PPC

    Important
    ---------
    Les noms de bandes PSD et PPC peuvent se recouvrir (ex: delta/theta/alpha/...),
    ce qui est normal. L'ambiguïté est levée par le scope :
    - all_channels / single_channel -> PSD
    - all_edges / single_edge -> PPC
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
        subject_set = frozenset(v.strip() for v in subject_variables)
        eeg_set = frozenset(v.strip() for v in eeg_features)
        psd_set = frozenset(v.strip() for v in psd_bands)
        ppc_set = frozenset(v.strip() for v in ppc_bands)

        # On interdit les collisions incompatibles, mais PAS PSD/PPC,
        # car elles sont résolues par le scope.
        overlaps = {
            "subject/eeg": subject_set & eeg_set,
            "subject/psd": subject_set & psd_set,
            "subject/ppc": subject_set & ppc_set,
            "eeg/psd": eeg_set & psd_set,
            "eeg/ppc": eeg_set & ppc_set,
        }

        duplicated_names = {
            family: sorted(values)
            for family, values in overlaps.items()
            if values
        }

        if duplicated_names:
            raise ValueError(
                "Some names overlap across incompatible query families: "
                f"{duplicated_names}"
            )

        return cls(
            subject_variables=subject_set,
            eeg_features=eeg_set,
            psd_bands=psd_set,
            ppc_bands=ppc_set,
        )


# ======================================================================================
#                                   QUERY FACTORY
# ======================================================================================

class QueryFactory:
    """
    Factory intelligent pour construire les bonnes requêtes statistiques.

    Familles reconnues
    ------------------
    - variables sujet-level
    - features EEG scalaires
    - bandes PSD
    - bandes PPC

    Principe important
    ------------------
    Les noms de bandes peuvent être identiques pour PSD et PPC.
    L'inférence doit donc se faire d'abord via le scope, puis via le nom.
    """

    _DEFAULT_SUBJECT_SCOPE: Final[Scope] = "subject"

    def __init__(self, config: QueryFactoryConfig):
        self.config = config

    # ------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------

    def compare_groups(
        self,
        *,
        target: str,
        group_col: str,
        group_a: str,
        group_b: str,
        test_kind: TestKind,
        scope: Scope,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
    ) -> GroupComparisonQuery:
        """
        Construit automatiquement une query de comparaison de groupes.
        """
        normalized_target = self._normalize_name(target)

        # ----------------------------------------------------------
        # Subject-level
        # ----------------------------------------------------------
        if scope == "subject":
            if self._is_subject_variable(normalized_target):
                self._validate_subject_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=normalized_target,
                )
                return SubjectGroupComparisonQuery(
                    variable=normalized_target,
                    group_col=group_col,
                    group_a=group_a,
                    group_b=group_b,
                    test_kind=test_kind,
                    scope=scope,
                )

            raise ValueError(
                f"'{normalized_target}' cannot be used with scope='subject'. "
                f"Known subject variables: {sorted(self.config.subject_variables)}."
            )

        # ----------------------------------------------------------
        # Channel-level -> EEG feature or PSD band
        # ----------------------------------------------------------
        if scope in {"single_channel", "all_channels"}:
            if self._is_eeg_feature(normalized_target):
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=normalized_target,
                    family="EEG feature",
                )
                return EEGFeatureGroupComparisonQuery(
                    feature=normalized_target,
                    group_col=group_col,
                    group_a=group_a,
                    group_b=group_b,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                )

            if self._is_psd_band(normalized_target):
                self._validate_channel_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=normalized_target,
                    family="PSD band",
                )
                return PSDBandGroupComparisonQuery(
                    band=normalized_target,
                    group_col=group_col,
                    group_a=group_a,
                    group_b=group_b,
                    test_kind=test_kind,
                    scope=scope,
                    channel=channel,
                )

            raise ValueError(
                f"'{normalized_target}' cannot be used with scope='{scope}'. "
                f"Known EEG features: {sorted(self.config.eeg_features)}. "
                f"Known PSD bands: {sorted(self.config.psd_bands)}."
            )

        # ----------------------------------------------------------
        # Edge-level -> PPC band
        # ----------------------------------------------------------
        if scope in {"single_edge", "all_edges"}:
            if self._is_ppc_band(normalized_target):
                self._validate_edge_scope(
                    scope=scope,
                    channel=channel,
                    edge=edge,
                    name=normalized_target,
                    family="PPC band",
                )
                return PPCBandGroupComparisonQuery(
                    band=normalized_target,
                    group_col=group_col,
                    group_a=group_a,
                    group_b=group_b,
                    test_kind=test_kind,
                    scope=scope,
                    edge=edge,
                )

            raise ValueError(
                f"'{normalized_target}' cannot be used with scope='{scope}'. "
                f"Known PPC bands: {sorted(self.config.ppc_bands)}."
            )

        raise ValueError(f"Unsupported scope: {scope}")

    def correlate(
        self,
        *,
        x: str,
        y: str,
        test_kind: TestKind,
        scope: Scope,
        channel: Optional[str] = None,
        edge: Optional[str] = None,
    ) -> CorrelationQuery:
        """
        Construit automatiquement une query de corrélation.

        Règles actuelles
        ----------------
        - sujet vs sujet -> SubjectCorrelationQuery
        - EEG vs sujet   -> EEGFeatureCorrelationQuery
        - sujet vs EEG   -> EEGFeatureCorrelationQuery
        - PSD vs sujet   -> PSDBandCorrelationQuery
        - sujet vs PSD   -> PSDBandCorrelationQuery
        - PPC vs sujet   -> PPCBandCorrelationQuery
        - sujet vs PPC   -> PPCBandCorrelationQuery
        """
        x_name = self._normalize_name(x)
        y_name = self._normalize_name(y)

        x_is_subject = self._is_subject_variable(x_name)
        y_is_subject = self._is_subject_variable(y_name)

        x_is_eeg = self._is_eeg_feature(x_name)
        y_is_eeg = self._is_eeg_feature(y_name)

        x_is_psd = self._is_psd_band(x_name)
        y_is_psd = self._is_psd_band(y_name)

        x_is_ppc = self._is_ppc_band(x_name)
        y_is_ppc = self._is_ppc_band(y_name)

        # ----------------------------------------------------------
        # Subject-level correlation
        # ----------------------------------------------------------
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
                )

            raise ValueError(
                "With scope='subject', both variables must be subject-level variables. "
                f"Known subject variables: {sorted(self.config.subject_variables)}."
            )

        # ----------------------------------------------------------
        # Channel-level correlation -> EEG or PSD vs subject
        # ----------------------------------------------------------
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
                )

            raise ValueError(
                "With scope='single_channel' or 'all_channels', supported correlations are "
                "only EEG-subject and PSD-subject."
            )

        # ----------------------------------------------------------
        # Edge-level correlation -> PPC vs subject
        # ----------------------------------------------------------
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
                )

            raise ValueError(
                "With scope='single_edge' or 'all_edges', supported correlations are "
                "only PPC-subject."
            )

        raise ValueError(f"Unsupported scope: {scope}")

    # ------------------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "QueryFactory":
        """
        Construit un factory avec une configuration par défaut raisonnable.
        """
        config = QueryFactoryConfig.from_lists(
            subject_variables={
                "age",
                "mmse",
                "education_years",
                "subject_mmse",
                "subject_age",
                "subject_health",
            },
            eeg_features={
                "theta_beta_ratio",
                "theta_alpha_ratio",
                "gamma_alpha_ratio",
                "delta_alpha_ratio",
                "spectral_power_ratio",
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
        """
        Construit un factory à partir des métadonnées du dataset.
        """
        return cls(
            config=QueryFactoryConfig.from_lists(
                subject_variables=subject_variables,
                eeg_features=eeg_features,
                psd_bands=psd_bands,
                ppc_bands=ppc_bands,
            )
        )

    # ------------------------------------------------------------------------------
    # Internal helpers
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
            raise ValueError(
                f"'{name}' is a {family}, so edge must be None."
            )

        if scope == "single_channel" and channel is None:
            raise ValueError(
                f"'{name}' requires a channel when scope='single_channel'."
            )

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
            raise ValueError(
                f"'{name}' is a {family}, so channel must be None."
            )

        if scope == "single_edge" and edge is None:
            raise ValueError(
                f"'{name}' requires an edge when scope='single_edge'."
            )

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