import torch
from dataclasses import dataclass

from rules.decision_rule import (
    SimpleDecisionRule,
    DecisionOperator,
    DecisionRule,
    DecisionRulesFactory
)
from rules.temperature import TemperatureFeatureMapping, TemperatureFeatureMappingFactory, SelectedFeaturesDataset
from prediction.decision_tree.base import TrainedDecisionTree

@dataclass(frozen=True)
class DifferentiableSimpleDecisionRule:
    feature_name: str
    feature_index: int
    threshold: float
    operator: DecisionOperator
    temperature: float

    def evaluate(self, x_feat: torch.Tensor) -> torch.Tensor:
        feature_values = x_feat[:, self.feature_index]

        threshold = torch.tensor(
            self.threshold,
            dtype=x_feat.dtype,
            device=x_feat.device,
        )

        temperature = torch.tensor(
            self.temperature,
            dtype=x_feat.dtype,
            device=x_feat.device,
        )

        if self.operator == DecisionOperator.LOWER_EQUAL:
            return torch.sigmoid((threshold - feature_values) / temperature)

        if self.operator == DecisionOperator.LOWER:
            return torch.sigmoid((threshold - feature_values) / temperature)

        if self.operator == DecisionOperator.GREATER_EQUAL:
            return torch.sigmoid((feature_values - threshold) / temperature)

        if self.operator == DecisionOperator.GREATER:
            return torch.sigmoid((feature_values - threshold) / temperature)

        raise ValueError(f"Unsupported operator: {self.operator}")

    
    def __str__(self) -> str:
        return (
            f"{self.feature_name} {self.operator.value} {self.threshold} "
            f"(tau={self.temperature:.6f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


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


@dataclass(frozen=True)
class DifferentiableDecisionRule:
    predicted_class: str
    prediction_probability: float  # = purity
    support: int
    score: float
    differentiable_rules: list[DifferentiableSimpleDecisionRule]

    def evaluate(self, x_feat: torch.Tensor) -> torch.Tensor:
        truth_degrees = [
            simple_rule.evaluate(x_feat)
            for simple_rule in self.differentiable_rules
        ]

        return torch.stack(truth_degrees, dim=0).min(dim=0).values

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
    def __init__(self, differentiable_decision_rules:list[DifferentiableDecisionRule], temperature_feature_mapping:TemperatureFeatureMapping):
        self.differentiable_decision_rules = differentiable_decision_rules
        self.temperature_feature_mapping = temperature_feature_mapping

    def evaluate(self, x_feat: torch.Tensor) -> list[torch.Tensor]:
        """
        Retourne la truth value de chaque règle.

        Returns
        -------
        list of tensors, chacun de shape [batch_size]
        """
        return [
            rule.evaluate(x_feat)
            for rule in self.differentiable_decision_rules
        ]


class DifferentiableDecisionRulesFactory:

    @staticmethod
    def _build_one(decision_rule:DecisionRule, temperature_feature_mapping: TemperatureFeatureMapping):
        differentiable_rules = [
            DifferentiableSimpleDecisionRuleFactory.build(
                simple_decision_rule=simple_rule,
                feature_index=temperature_feature_mapping.index(simple_rule.feature_name),
                temperature=temperature_feature_mapping(simple_rule.feature_name),
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
    def build(trained_tree:TrainedDecisionTree, c_tau:float=0.1, min_tau:float=0.001) -> DifferentiableDecisionRules:

        temperature_feature_mapping = TemperatureFeatureMappingFactory.build(trained_tree.dataset, c_tau, min_tau)
        decision_rules = DecisionRulesFactory.build(trained_tree)

        differentiable_decision_rules = [DifferentiableDecisionRulesFactory._build_one(decision_rule, temperature_feature_mapping) for decision_rule in decision_rules]

        return DifferentiableDecisionRules(differentiable_decision_rules, temperature_feature_mapping)

        