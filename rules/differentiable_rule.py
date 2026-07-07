from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

import torch

from prediction.decision_tree.base import TrainedDecisionTree
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


class DifferentiableRule(ABC):
    """Base class for all differentiable rules."""

    pass


@dataclass(frozen=True)
class DifferentiableCondition(DifferentiableRule):
    """Differentiable version of a threshold-based decision-tree condition."""

    feature_name: str
    feature_index: int
    threshold: float
    operator: DecisionOperator
    temperature: float

    def __str__(self) -> str:
        """Return a human-readable differentiable condition."""
        return (
            f"{self.feature_name} {self.operator.value} {self.threshold} "
            f"(tau={self.temperature:.6f})"
        )

    def __repr__(self) -> str:
        """Return the string representation of the differentiable condition."""
        return self.__str__()


@dataclass(frozen=True)
class DifferentiableDecisionRule(DifferentiableRule):
    """Differentiable version of a decision rule."""

    predicted_class: str
    prediction_probability: float
    support: int
    score: float
    differentiable_conditions: list[DifferentiableCondition]

    def __str__(self) -> str:
        """Return a human-readable differentiable decision rule."""
        conds = "\n  AND ".join(str(rule) for rule in self.differentiable_conditions)

        return (
            f"{conds}\n"
            f"=> class={self.predicted_class} "
            f"(n={self.support}, p={self.prediction_probability:.3f}, "
            f"score={self.score:.3f})"
        )

    def __repr__(self) -> str:
        """Return the string representation of the differentiable decision rule."""
        return self.__str__()


class DifferentiableDecisionRules:
    """Container for differentiable decision rules and their temperature mapping."""

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
    PyTorch-friendly candidate batch.

    ``x_feat`` must be a tensor with shape ``[batch_size, n_features]``.
    """

    x_feat: torch.Tensor

    def __post_init__(self) -> None:
        """Validate the candidate feature tensor."""
        if not isinstance(self.x_feat, torch.Tensor):
            raise TypeError("x_feat must be a torch.Tensor.")

        if self.x_feat.ndim != 2:
            raise ValueError(
                "DifferentiableRuleCandidate expects x_feat "
                "with shape [batch_size, n_features]."
            )

    @property
    def batch_size(self) -> int:
        """Return the batch size."""
        return int(self.x_feat.shape[0])

    @property
    def n_features(self) -> int:
        """Return the number of features."""
        return int(self.x_feat.shape[1])

    @property
    def dtype(self) -> torch.dtype:
        """Return the tensor dtype."""
        return self.x_feat.dtype

    @property
    def device(self) -> torch.device:
        """Return the tensor device."""
        return self.x_feat.device

    def get_feature_values(self, feature_index: int) -> torch.Tensor:
        """Return the batch values for one feature index."""
        if feature_index < 0 or feature_index >= self.n_features:
            raise IndexError(
                f"Invalid feature_index={feature_index}. "
                f"Candidate has {self.n_features} features."
            )

        return self.x_feat[:, feature_index]


class DifferentiableRuleCandidateFactory:
    """Factory used to build differentiable rule candidates."""

    @staticmethod
    def from_tensor(x_feat: torch.Tensor) -> DifferentiableRuleCandidate:
        """Build a candidate batch from a feature tensor."""
        return DifferentiableRuleCandidate(x_feat=x_feat)

    @staticmethod
    def from_single_candidate(
        x_feat: torch.Tensor,
    ) -> DifferentiableRuleCandidate:
        """Build a one-sample candidate batch from a one-dimensional tensor."""
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
    Engine computing differentiable truth degrees.

    Outputs
    -------
    - DifferentiableCondition    -> Tensor with shape ``[batch_size]``
    - DifferentiableDecisionRule -> Tensor with shape ``[batch_size]``
    """

    @staticmethod
    def compute(
        rule: DifferentiableRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        """Compute the differentiable truth degree of a rule for a candidate batch."""
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
        """Compute the differentiable truth degree of one condition."""
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

        sign = (
            1
            if condition.operator in {DecisionOperator.LOWER, DecisionOperator.LOWER_EQUAL}
            else -1
        )

        return torch.sigmoid(sign * (threshold - feature_values) / temperature)

    @staticmethod
    def _compute_decision_rule(
        rule: DifferentiableDecisionRule,
        candidate: DifferentiableRuleCandidate,
    ) -> torch.Tensor:
        """Compute the differentiable truth degree of a full decision rule."""
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
    """Factory used to convert a condition into a differentiable condition."""

    @staticmethod
    def build(
        simple_decision_rule: Condition,
        feature_index: int,
        temperature: float,
    ) -> DifferentiableCondition:
        """Build a differentiable condition from a standard condition."""
        return DifferentiableCondition(
            feature_name=simple_decision_rule.feature_name,
            feature_index=feature_index,
            threshold=simple_decision_rule.threshold,
            operator=simple_decision_rule.operator,
            temperature=temperature,
        )


class DifferentiableDecisionRulesFactory:
    """Factory used to build differentiable decision rules from a trained tree."""

    @staticmethod
    def _build_one(
        decision_rule: DecisionRule,
        temperature_feature_mapping: TemperatureFeatureMapping,
    ) -> DifferentiableDecisionRule:
        """Build one differentiable decision rule from one standard decision rule."""
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
        """Build all differentiable decision rules from a trained decision tree."""
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