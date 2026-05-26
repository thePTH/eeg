from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Union

import torch

from rules.decision_rule import (
    Condition,
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
class DifferentiableCondition(DifferentiableRule):
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
    differentiable_conditions: list[DifferentiableCondition]

    def __str__(self) -> str:
        conds = "\n  AND ".join(str(rule) for rule in self.differentiable_conditions)

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





class TruthDegreeEngine:
    """
    Calcule des degrés de vérité différentiables.

    Sorties
    -------
    - DifferentiableSimpleDecisionRule -> Tensor [batch_size]
    - DifferentiableDecisionRule       -> Tensor [batch_size]
    
    """

    @staticmethod
    def compute(
        rule: DifferentiableRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        if isinstance(rule, DifferentiableCondition):
            return TruthDegreeEngine._compute_condition(
                condition=rule,
                candidate=candidate,
            )

        if isinstance(rule, DifferentiableDecisionRule):
            return TruthDegreeEngine._compute_decision_rule(
                rule=rule,
                candidate=candidate,
            )

        

        raise TypeError(f"Unsupported rules type: {type(rule).__name__}")

    @staticmethod
    def _compute_condition(
        condition: DifferentiableCondition,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        feature_values = candidate.get_feature_values(condition.feature_index)

        threshold = torch.tensor(
            condition.threshold,
            dtype=candidate.dtype,
            device=candidate.device,
        )

        temperature = torch.tensor(
            condition.temperature,
            dtype=candidate.dtype,
            device=candidate.device,
        )

        sign = 1 if condition.operator in {DecisionOperator.LOWER, DecisionOperator.LOWER_EQUAL} else - 1


        return torch.sigmoid(sign * (threshold - feature_values) / temperature)

        

    @staticmethod
    def _compute_decision_rule(
        rule: DifferentiableDecisionRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        if len(rule.differentiable_conditions) == 0:
            return torch.ones(
                candidate.batch_size,
                dtype=candidate.dtype,
                device=candidate.device,
            )

        truth_degrees = [
            TruthDegreeEngine.compute(
                rule=condition,
                candidate=candidate,
            )
            for condition in rule.differentiable_conditions
        ]

        return torch.stack(truth_degrees, dim=0).min(dim=0).values

    


class DifferentiableSimpleDecisionRuleFactory:
    @staticmethod
    def build(
        simple_decision_rule: Condition,
        feature_index: int,
        temperature: float,
    ) -> DifferentiableCondition:
        return DifferentiableCondition(
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
            for simple_rule in decision_rule.conditions
        ]

        return DifferentiableDecisionRule(
            predicted_class=decision_rule.predicted_class,
            prediction_probability=decision_rule.prediction_probability,
            support=decision_rule.support,
            score=decision_rule.score,
            differentiable_conditions=differentiable_rules,
        )

    @staticmethod
    def build(
        trained_tree: TrainedDecisionTree,
        c_tau: float = 0.1,
        min_tau: float = 0.001,
    ) -> tuple[list[DifferentiableDecisionRule], TemperatureFeatureMapping]:
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

        return differentiable_decision_rules, temperature_feature_mapping