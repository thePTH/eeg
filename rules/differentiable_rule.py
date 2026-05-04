from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Union

import torch

from rules.decision_rule import (
    SimpleDecisionRule,
    DecisionOperator,
    DecisionRule,
    DecisionRulesFactory,
)
from rules.temperature import (
    TemperatureFeatureMapping,
    TemperatureFeatureMappingFactory,
)
from prediction.decision_tree.base import TrainedDecisionTree


class DifferentiableRule(ABC):
    """Classe mère de toutes les règles différentiables."""
    pass


@dataclass(frozen=True)
class DifferentiableSimpleDecisionRule(DifferentiableRule):
    feature_name: str
    feature_index: int
    threshold: float
    operator: DecisionOperator
    temperature: float

    def __str__(self) -> str:
        return (
            f"{self.feature_name} {self.operator.value} {self.threshold} "
            f"(tau={self.temperature:.6f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class DifferentiableDecisionRule(DifferentiableRule):
    predicted_class: str
    prediction_probability: float
    support: int
    score: float
    differentiable_rules: list[DifferentiableSimpleDecisionRule]

    def __str__(self) -> str:
        conds = "\n  AND ".join(str(rule) for rule in self.differentiable_rules)

        return (
            f"{conds}\n"
            f"=> class={self.predicted_class} "
            f"(n={self.support}, p={self.prediction_probability:.3f}, "
            f"score={self.score:.3f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class DifferentiableDecisionRules:
    def __init__(
        self,
        differentiable_decision_rules: list[DifferentiableDecisionRule],
        temperature_feature_mapping: TemperatureFeatureMapping,
    ) -> None:
        self.differentiable_decision_rules = differentiable_decision_rules
        self.temperature_feature_mapping = temperature_feature_mapping


@dataclass(frozen=True)
class DifferentiableRuleCandidate:
    """
    Candidat PyTorch-friendly.

    x_feat doit être un tensor de shape [batch_size, n_features].
    """

    x_feat: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.x_feat, torch.Tensor):
            raise TypeError("x_feat must be a torch.Tensor.")

        if self.x_feat.ndim != 2:
            raise ValueError(
                "DifferentiableRuleCandidate expects x_feat "
                "with shape [batch_size, n_features]."
            )

    @property
    def batch_size(self) -> int:
        return int(self.x_feat.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.x_feat.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        return self.x_feat.dtype

    @property
    def device(self) -> torch.device:
        return self.x_feat.device

    def get_feature_values(self, feature_index: int) -> torch.Tensor:
        if feature_index < 0 or feature_index >= self.n_features:
            raise IndexError(
                f"Invalid feature_index={feature_index}. "
                f"Candidate has {self.n_features} features."
            )

        return self.x_feat[:, feature_index]


class DifferentiableRuleCandidateFactory:
    @staticmethod
    def from_tensor(x_feat: torch.Tensor) -> DifferentiableRuleCandidate:
        return DifferentiableRuleCandidate(x_feat=x_feat)

    @staticmethod
    def from_single_candidate(
        x_feat: torch.Tensor,
    ) -> DifferentiableRuleCandidate:
        if not isinstance(x_feat, torch.Tensor):
            raise TypeError("x_feat must be a torch.Tensor.")

        if x_feat.ndim != 1:
            raise ValueError(
                "from_single_candidate expects a tensor "
                "with shape [n_features]."
            )

        return DifferentiableRuleCandidate(
            x_feat=x_feat.unsqueeze(0)
        )


DifferentiableRuleInput = Union[
    DifferentiableRule,
    list[DifferentiableRule],
    DifferentiableDecisionRules,
]


class TruthDegreeEngine:
    """
    Calcule des degrés de vérité différentiables.

    Sorties
    -------
    - DifferentiableSimpleDecisionRule -> Tensor [batch_size]
    - DifferentiableDecisionRule       -> Tensor [batch_size]
    - list[DifferentiableRule]         -> Tensor [batch_size, n_rules]
    - DifferentiableDecisionRules      -> Tensor [batch_size, n_rules]
    """

    @staticmethod
    def compute(
        rules: DifferentiableRuleInput,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        if isinstance(rules, DifferentiableSimpleDecisionRule):
            return TruthDegreeEngine._compute_simple_rule(
                rule=rules,
                candidate=candidate,
            )

        if isinstance(rules, DifferentiableDecisionRule):
            return TruthDegreeEngine._compute_decision_rule(
                rule=rules,
                candidate=candidate,
            )

        if isinstance(rules, DifferentiableDecisionRules):
            return TruthDegreeEngine._compute_decision_rules(
                rules=rules,
                candidate=candidate,
            )

        if isinstance(rules, list):
            return TruthDegreeEngine._compute_rule_list(
                rules=rules,
                candidate=candidate,
            )

        raise TypeError(f"Unsupported rules type: {type(rules).__name__}")

    @staticmethod
    def _compute_simple_rule(
        rule: DifferentiableSimpleDecisionRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        feature_values = candidate.get_feature_values(rule.feature_index)

        threshold = torch.tensor(
            rule.threshold,
            dtype=candidate.dtype,
            device=candidate.device,
        )

        temperature = torch.tensor(
            rule.temperature,
            dtype=candidate.dtype,
            device=candidate.device,
        )

        if rule.operator in {
            DecisionOperator.LOWER,
            DecisionOperator.LOWER_EQUAL,
        }:
            return torch.sigmoid(
                (threshold - feature_values) / temperature
            )

        if rule.operator in {
            DecisionOperator.GREATER,
            DecisionOperator.GREATER_EQUAL,
        }:
            return torch.sigmoid(
                (feature_values - threshold) / temperature
            )

        raise ValueError(f"Unsupported operator: {rule.operator}")

    @staticmethod
    def _compute_decision_rule(
        rule: DifferentiableDecisionRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        if len(rule.differentiable_rules) == 0:
            return torch.ones(
                candidate.batch_size,
                dtype=candidate.dtype,
                device=candidate.device,
            )

        truth_degrees = [
            TruthDegreeEngine.compute(
                rules=simple_rule,
                candidate=candidate,
            )
            for simple_rule in rule.differentiable_rules
        ]

        return torch.stack(truth_degrees, dim=0).min(dim=0).values

    @staticmethod
    def _compute_decision_rules(
        rules: DifferentiableDecisionRules,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        return TruthDegreeEngine._compute_rule_list(
            rules=rules.differentiable_decision_rules,
            candidate=candidate,
        )

    @staticmethod
    def _compute_rule_list(
        rules: list[DifferentiableRule],
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        if len(rules) == 0:
            return torch.empty(
                candidate.batch_size,
                0,
                dtype=candidate.dtype,
                device=candidate.device,
            )

        truth_degrees = [
            TruthDegreeEngine.compute(
                rules=rule,
                candidate=candidate,
            )
            for rule in rules
        ]

        return torch.stack(truth_degrees, dim=1)


class DifferentiableSimpleDecisionRuleFactory:
    @staticmethod
    def build(
        simple_decision_rule: SimpleDecisionRule,
        feature_index: int,
        temperature: float,
    ) -> DifferentiableSimpleDecisionRule:
        return DifferentiableSimpleDecisionRule(
            feature_name=simple_decision_rule.feature_name,
            feature_index=feature_index,
            threshold=simple_decision_rule.threshold,
            operator=simple_decision_rule.operator,
            temperature=temperature,
        )


class DifferentiableDecisionRulesFactory:
    @staticmethod
    def _build_one(
        decision_rule: DecisionRule,
        temperature_feature_mapping: TemperatureFeatureMapping,
    ) -> DifferentiableDecisionRule:
        differentiable_rules = [
            DifferentiableSimpleDecisionRuleFactory.build(
                simple_decision_rule=simple_rule,
                feature_index=temperature_feature_mapping.index(
                    simple_rule.feature_name
                ),
                temperature=temperature_feature_mapping(
                    simple_rule.feature_name
                ),
            )
            for simple_rule in decision_rule.simple_rules
        ]

        return DifferentiableDecisionRule(
            predicted_class=decision_rule.predicted_class,
            prediction_probability=decision_rule.prediction_probability,
            support=decision_rule.support,
            score=decision_rule.score,
            differentiable_rules=differentiable_rules,
        )

    @staticmethod
    def build(
        trained_tree: TrainedDecisionTree,
        c_tau: float = 0.1,
        min_tau: float = 0.001,
    ) -> DifferentiableDecisionRules:
        temperature_feature_mapping = TemperatureFeatureMappingFactory.build(
            trained_tree.dataset,
            c_tau,
            min_tau,
        )

        decision_rules = DecisionRulesFactory.build(trained_tree)

        differentiable_decision_rules = [
            DifferentiableDecisionRulesFactory._build_one(
                decision_rule=decision_rule,
                temperature_feature_mapping=temperature_feature_mapping,
            )
            for decision_rule in decision_rules
        ]

        return DifferentiableDecisionRules(
            differentiable_decision_rules=differentiable_decision_rules,
            temperature_feature_mapping=temperature_feature_mapping,
        )