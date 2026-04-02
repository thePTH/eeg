from __future__ import annotations

from dataclasses import dataclass
from abc import ABC
from typing import Optional, Literal


Scope = Literal["subject", "single_channel", "all_channels"]
TestKind = Literal["t_test", "wilcoxon_rank_sum", "spearman"]


@dataclass(frozen=True)
class StatisticalQuery(ABC):
    """
    Classe abstraite racine pour toutes les requêtes statistiques.

    Une query porte l'intention métier :
    - quelle variable / feature ?
    - quelle portée ?
    - quel test statistique ?
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


@dataclass(frozen=True)
class SubjectGroupComparisonQuery(GroupComparisonQuery):
    """
    Exemples:
    - âge Healthy vs Alzheimer
    - MMSE Healthy vs Alzheimer
    - années d'éducation Healthy vs Alzheimer
    """
    variable: str

    @property
    def target_name(self) -> str:
        return self.variable


@dataclass(frozen=True)
class EEGFeatureGroupComparisonQuery(GroupComparisonQuery):
    """
    Exemples:
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
        if self.scope != "single_channel" and self.channel is not None:
            raise ValueError("channel must be None unless scope='single_channel'")


@dataclass(frozen=True)
class SubjectCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre deux variables sujet-level.
    Exemple:
    - age vs mmse
    """
    x_variable: str
    y_variable: str

    @property
    def target_name(self) -> str:
        return self.x_variable


@dataclass(frozen=True)
class EEGFeatureCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une feature EEG et une covariable sujet-level.
    Exemples:
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
        if self.scope != "single_channel" and self.channel is not None:
            raise ValueError("channel must be None unless scope='single_channel'")
        



from dataclasses import dataclass
from abc import ABC
from typing import Optional, Literal, Iterable, Final


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
    - Avec quelle portée (niveau sujet, canal unique, tous les canaux) ?
    - Avec quel test statistique ?

    Attributs
    ---------
    test_kind:
        Type de test statistique à appliquer.
        Exemples : "t_test", "wilcoxon_rank_sum", "spearman".

    scope:
        Portée de la requête.
        - "subject" : niveau sujet
        - "single_channel" : un seul canal EEG
        - "all_channels" : tous les canaux EEG
    """
    test_kind: TestKind
    scope: Scope

    @property
    def target_name(self) -> str:
        """
        Nom métier principal de la cible testée.

        Cette propriété est utile pour :
        - le logging
        - l'affichage UI
        - les titres de résultats
        - les exports

        Returns
        -------
        str
            Nom principal de la cible.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class GroupComparisonQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour comparer deux groupes.

    Exemples :
    - comparer l'âge entre Healthy et Alzheimer
    - comparer une feature EEG entre Prodromal et Alzheimer

    Attributs
    ---------
    group_col:
        Nom de la colonne qui porte l'appartenance au groupe
        (ex: "diagnosis", "group", "label").

    group_a:
        Première modalité / premier groupe.

    group_b:
        Seconde modalité / second groupe.
    """
    group_col: str
    group_a: str
    group_b: str


@dataclass(frozen=True)
class CorrelationQuery(StatisticalQuery, ABC):
    """
    Requête abstraite pour corrélation entre deux variables.

    Exemples :
    - âge vs MMSE
    - theta_alpha_ratio vs subject_mmse
    """
    pass


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
class EEGFeatureGroupComparisonQuery(GroupComparisonQuery):
    """
    Comparaison de groupes sur une feature EEG.

    Exemples
    --------
    - theta_beta_ratio Healthy vs Alzheimer sur tous les canaux
    - gamma_alpha_ratio Healthy vs Prodromal sur un canal donné

    Attributs
    ---------
    feature:
        Nom de la feature EEG.

    channel:
        Nom du canal EEG si la portée est "single_channel".
        Doit être None sinon.
    """
    feature: str
    channel: Optional[str] = None

    @property
    def target_name(self) -> str:
        return self.feature

    def __post_init__(self):
        if self.scope == "single_channel" and self.channel is None:
            raise ValueError("channel must be provided when scope='single_channel'")
        if self.scope != "single_channel" and self.channel is not None:
            raise ValueError("channel must be None unless scope='single_channel'")


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


@dataclass(frozen=True)
class EEGFeatureCorrelationQuery(CorrelationQuery):
    """
    Corrélation entre une feature EEG et une covariable sujet-level.

    Exemples
    --------
    - theta_alpha_ratio vs subject_mmse sur tous les canaux

    Attributs
    ---------
    feature:
        Nom de la feature EEG.

    covariate:
        Variable sujet-level corrélée à la feature EEG.

    channel:
        Canal EEG si scope == "single_channel", sinon None.
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
        if self.scope != "single_channel" and self.channel is not None:
            raise ValueError("channel must be None unless scope='single_channel'")


# ======================================================================================
#                               QUERY FACTORY CONFIG
# ======================================================================================

@dataclass(frozen=True)
class QueryFactoryConfig:
    """
    Configuration du QueryFactory.

    Cette configuration sert à indiquer au factory comment distinguer :
    - les variables sujet-level
    - les features EEG

    Pourquoi une config ?
    ---------------------
    Le factory doit prendre des décisions d'inférence. Pour cela, il doit connaître
    le "vocabulaire" valide du domaine métier.

    Exemple :
    ---------
    QueryFactoryConfig(
        subject_variables={"age", "mmse", "education_years", "subject_mmse"},
        eeg_features={"theta_beta_ratio", "theta_alpha_ratio", "gamma_alpha_ratio"},
    )
    """
    subject_variables: frozenset[str]
    eeg_features: frozenset[str]

    @classmethod
    def from_lists(
        cls,
        *,
        subject_variables: Iterable[str],
        eeg_features: Iterable[str],
    ) -> "QueryFactoryConfig":
        """
        Construit une configuration à partir d'itérables de chaînes.

        Tous les noms sont normalisés via `strip()`.

        Parameters
        ----------
        subject_variables:
            Variables sujet-level connues.

        eeg_features:
            Features EEG connues.

        Returns
        -------
        QueryFactoryConfig
            Configuration prête à l'emploi.
        """
        subject_set = frozenset(v.strip() for v in subject_variables)
        eeg_set = frozenset(v.strip() for v in eeg_features)

        overlap = subject_set & eeg_set
        if overlap:
            raise ValueError(
                "Some names are present both in subject_variables and eeg_features: "
                f"{sorted(overlap)}"
            )

        return cls(
            subject_variables=subject_set,
            eeg_features=eeg_set,
        )


# ======================================================================================
#                                   QUERY FACTORY
# ======================================================================================

class QueryFactory:
    """
    Factory intelligent pour construire les bonnes requêtes statistiques.

    Philosophie
    -----------
    L'objectif est d'offrir une API très simple côté appelant :
    - `compare_groups(...)`
    - `correlate(...)`

    puis de laisser le factory :
    - inférer le bon type de Query
    - valider les combinaisons d'arguments
    - lever des erreurs explicites si l'intention métier est ambiguë ou invalide

    Stratégie d'inférence
    ---------------------
    1. Comparaison de groupes :
       - si `target` est une variable sujet connue -> SubjectGroupComparisonQuery
       - si `target` est une feature EEG connue -> EEGFeatureGroupComparisonQuery

    2. Corrélation :
       - si `x` et `y` sont deux variables sujet connues -> SubjectCorrelationQuery
       - si l'un est une feature EEG et l'autre une variable sujet -> EEGFeatureCorrelationQuery
       - si les deux sont des features EEG -> non supporté dans cette version
         (car le modèle métier actuel ne définit pas une query dédiée)

    Notes de conception
    -------------------
    - Le factory ne lance pas le test statistique. Il ne fait que construire une
      représentation métier correcte.
    - La distinction sujet / EEG repose sur une configuration explicite pour garder
      une logique robuste et prédictible.
    """

    _DEFAULT_SUBJECT_SCOPE: Final[Scope] = "subject"

    def __init__(self, config: QueryFactoryConfig):
        """
        Initialise le factory.

        Parameters
        ----------
        config:
            Configuration contenant les variables sujet et les features EEG reconnues.
        """
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
    ) -> GroupComparisonQuery:
        """
        Construit automatiquement une query de comparaison de groupes.

        Parameters
        ----------
        target:
            Variable métier à comparer entre deux groupes.
            Peut être :
            - une variable sujet-level (ex: "age", "mmse")
            - une feature EEG (ex: "theta_beta_ratio")

        group_col:
            Colonne d'appartenance au groupe
            (ex: "diagnosis").

        group_a:
            Premier groupe.

        group_b:
            Second groupe.

        test_kind:
            Test statistique à utiliser.
            Exemples : "t_test", "wilcoxon_rank_sum".

        scope:
            Portée de l'analyse.
            - "subject" pour variable sujet
            - "single_channel" ou "all_channels" pour EEG
            - "subject" peut aussi être utilisé pour des features non EEG uniquement
              si ton métier le permet, mais ici on l'interprète strictement

        channel:
            Canal EEG si `scope == "single_channel"`.

        Returns
        -------
        GroupComparisonQuery
            Une instance concrète de :
            - SubjectGroupComparisonQuery
            - EEGFeatureGroupComparisonQuery

        Raises
        ------
        ValueError
            Si la variable est inconnue ou si les arguments sont incohérents.
        """
        normalized_target = self._normalize_name(target)

        if self._is_subject_variable(normalized_target):
            self._validate_subject_scope(scope=scope, channel=channel, name=normalized_target)

            return SubjectGroupComparisonQuery(
                variable=normalized_target,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                test_kind=test_kind,
                scope=scope,
            )

        if self._is_eeg_feature(normalized_target):
            self._validate_eeg_scope(scope=scope, channel=channel, name=normalized_target)

            return EEGFeatureGroupComparisonQuery(
                feature=normalized_target,
                group_col=group_col,
                group_a=group_a,
                group_b=group_b,
                test_kind=test_kind,
                scope=scope,
                channel=channel,
            )

        raise ValueError(self._unknown_variable_message(normalized_target))

    def correlate(
        self,
        *,
        x: str,
        y: str,
        test_kind: TestKind,
        scope: Scope,
        channel: Optional[str] = None,
    ) -> CorrelationQuery:
        """
        Construit automatiquement une query de corrélation.

        Règles actuelles
        ----------------
        - sujet vs sujet -> SubjectCorrelationQuery
        - EEG vs sujet   -> EEGFeatureCorrelationQuery
        - sujet vs EEG   -> EEGFeatureCorrelationQuery (avec permutation interne)
        - EEG vs EEG     -> non supporté dans cette version

        Parameters
        ----------
        x:
            Première variable.

        y:
            Seconde variable.

        test_kind:
            Type de test, généralement "spearman" pour la corrélation.

        scope:
            Portée de l'analyse.
            - "subject" pour sujet vs sujet
            - "single_channel" / "all_channels" pour EEG vs sujet

        channel:
            Canal EEG si `scope == "single_channel"`.

        Returns
        -------
        CorrelationQuery
            Une instance concrète de :
            - SubjectCorrelationQuery
            - EEGFeatureCorrelationQuery

        Raises
        ------
        ValueError
            Si les noms sont inconnus, ambigus, ou si la combinaison n'est pas supportée.
        """
        x_name = self._normalize_name(x)
        y_name = self._normalize_name(y)

        x_is_subject = self._is_subject_variable(x_name)
        y_is_subject = self._is_subject_variable(y_name)
        x_is_eeg = self._is_eeg_feature(x_name)
        y_is_eeg = self._is_eeg_feature(y_name)

        # Cas 1 : sujet vs sujet
        if x_is_subject and y_is_subject:
            self._validate_subject_scope(scope=scope, channel=channel, name=f"{x_name} vs {y_name}")

            return SubjectCorrelationQuery(
                x_variable=x_name,
                y_variable=y_name,
                test_kind=test_kind,
                scope=scope,
            )

        # Cas 2 : EEG vs sujet
        if x_is_eeg and y_is_subject:
            self._validate_eeg_scope(scope=scope, channel=channel, name=f"{x_name} vs {y_name}")

            return EEGFeatureCorrelationQuery(
                feature=x_name,
                covariate=y_name,
                test_kind=test_kind,
                scope=scope,
                channel=channel,
            )

        # Cas 3 : sujet vs EEG -> on inverse pour respecter le modèle feature/covariate
        if x_is_subject and y_is_eeg:
            self._validate_eeg_scope(scope=scope, channel=channel, name=f"{x_name} vs {y_name}")

            return EEGFeatureCorrelationQuery(
                feature=y_name,
                covariate=x_name,
                test_kind=test_kind,
                scope=scope,
                channel=channel,
            )

        # Cas 4 : EEG vs EEG -> non défini dans ton modèle actuel
        if x_is_eeg and y_is_eeg:
            raise ValueError(
                "Correlation between two EEG features is not supported by the current "
                "query model. Define a dedicated query class if needed."
            )

        # Cas 5 : variable(s) inconnue(s)
        unknown_names = []
        if not (x_is_subject or x_is_eeg):
            unknown_names.append(x_name)
        if not (y_is_subject or y_is_eeg):
            unknown_names.append(y_name)

        raise ValueError(
            "Unknown variable name(s): "
            f"{unknown_names}. "
            f"Known subject variables: {sorted(self.config.subject_variables)}. "
            f"Known EEG features: {sorted(self.config.eeg_features)}."
        )

    # ------------------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------------------

    @classmethod
    def with_defaults(cls) -> "QueryFactory":
        """
        Construit un factory avec une configuration par défaut raisonnable.

        Cette méthode est pratique pour démarrer vite, faire des tests, ou produire
        des notebooks plus lisibles.

        Returns
        -------
        QueryFactory
            Factory préconfiguré.
        """
        config = QueryFactoryConfig.from_lists(
            subject_variables={
                "age",
                "mmse",
                "education_years",
                "subject_mmse",
                "subject_age",
            },
            eeg_features={
                "theta_beta_ratio",
                "theta_alpha_ratio",
                "gamma_alpha_ratio",
                "delta_alpha_ratio",
            },
        )
        return cls(config=config)

    # ------------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        """
        Normalise un nom de variable.

        Actuellement, on applique seulement `strip()` pour rester prudent.
        On peut enrichir plus tard :
        - lower()
        - mapping d'alias
        - synonymes métier

        Parameters
        ----------
        name:
            Nom brut.

        Returns
        -------
        str
            Nom normalisé.

        Raises
        ------
        ValueError
            Si la chaîne est vide après normalisation.
        """
        normalized = name.strip()
        if not normalized:
            raise ValueError("Variable name cannot be empty.")
        return normalized

    def _is_subject_variable(self, name: str) -> bool:
        """
        Indique si un nom correspond à une variable sujet connue.
        """
        return name in self.config.subject_variables

    def _is_eeg_feature(self, name: str) -> bool:
        """
        Indique si un nom correspond à une feature EEG connue.
        """
        return name in self.config.eeg_features

    def _validate_subject_scope(
        self,
        *,
        scope: Scope,
        channel: Optional[str],
        name: str,
    ) -> None:
        """
        Valide qu'une variable sujet est utilisée avec une portée cohérente.

        Règle :
        - une variable sujet doit être utilisée avec `scope="subject"`
        - `channel` doit être None
        """
        if scope != "subject":
            raise ValueError(
                f"'{name}' is a subject-level variable and therefore requires "
                "scope='subject'."
            )
        if channel is not None:
            raise ValueError(
                f"'{name}' is a subject-level variable, so channel must be None."
            )

    def _validate_eeg_scope(
        self,
        *,
        scope: Scope,
        channel: Optional[str],
        name: str,
    ) -> None:
        """
        Valide qu'une feature EEG est utilisée avec une portée cohérente.

        Règles :
        - EEG ne peut pas avoir scope='subject'
        - si scope='single_channel', channel doit être fourni
        - sinon channel doit être None
        """
        if scope == "subject":
            raise ValueError(
                f"'{name}' is an EEG feature and cannot be used with scope='subject'. "
                "Use scope='single_channel' or scope='all_channels'."
            )

        if scope == "single_channel" and channel is None:
            raise ValueError(
                f"'{name}' requires a channel when scope='single_channel'."
            )

        if scope != "single_channel" and channel is not None:
            raise ValueError(
                f"'{name}' received channel='{channel}', but channel is only allowed "
                "when scope='single_channel'."
            )

    def _unknown_variable_message(self, name: str) -> str:
        """
        Construit un message d'erreur détaillé pour une variable inconnue.
        """
        return (
            f"Unknown variable '{name}'. "
            f"Known subject variables: {sorted(self.config.subject_variables)}. "
            f"Known EEG features: {sorted(self.config.eeg_features)}."
        )