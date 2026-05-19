from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import _tree

from prediction.decision_tree.base import TrainedDecisionTree


class DecisionOperator(str, Enum):
    GREATER = ">"
    LOWER = "<"
    GREATER_EQUAL = ">="
    LOWER_EQUAL = "<="

    def apply(self, left_value: float, right_value: float) -> bool:
        match self:
            case DecisionOperator.GREATER:
                return left_value > right_value
            case DecisionOperator.LOWER:
                return left_value < right_value
            case DecisionOperator.GREATER_EQUAL:
                return left_value >= right_value
            case DecisionOperator.LOWER_EQUAL:
                return left_value <= right_value
            case _:
                raise ValueError(f"Unsupported operator: {self}")


class Rule(ABC):
    """Classe mère de toutes les règles évaluables."""
    pass


@dataclass(frozen=True)
class Condition(Rule):
    feature_name: str
    threshold: float
    operator: DecisionOperator

    def __str__(self) -> str:
        return f"{self.feature_name} {self.operator.value} {self.threshold}"


@dataclass
class DecisionRule(Rule):
    predicted_class: str
    prediction_probability: float
    support: int
    score: float
    conditions: list[Condition]

    def __str__(self) -> str:
        conds = "\n  AND ".join(str(rule) for rule in self.conditions)

        return (
            f"{conds}\n"
            f"=> class={self.predicted_class} "
            f"(n={self.support}, p={self.prediction_probability:.3f}, "
            f"score={self.score:.3f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


@dataclass(frozen=True)
class RuleCheckerCandidate:
    values: dict[str, Any]

    def get_value(self, feature_name: str) -> Any:
        if feature_name not in self.values:
            raise KeyError(f"Missing feature in candidate: {feature_name}")

        return self.values[feature_name]


class RuleCheckerCandidateFactory:
    @staticmethod
    def from_dataframe(candidate: pd.DataFrame) -> RuleCheckerCandidate:
        if len(candidate) != 1:
            raise ValueError(
                "RuleCheckerCandidateFactory.from_dataframe expects "
                "a DataFrame containing exactly one row."
            )

        return RuleCheckerCandidate(
            values=candidate.iloc[0].to_dict()
        )

    @staticmethod
    def from_series(candidate: pd.Series) -> RuleCheckerCandidate:
        return RuleCheckerCandidate(
            values=candidate.to_dict()
        )

    @staticmethod
    def from_dict(candidate: dict[str, Any]) -> RuleCheckerCandidate:
        return RuleCheckerCandidate(
            values=dict(candidate)
        )


class DecisionRuleChecker:
    @staticmethod
    def check(rule: Rule, candidate: RuleCheckerCandidate) -> bool:
        if isinstance(rule, Condition):
            return DecisionRuleChecker._check_condition(
                rule=rule,
                candidate=candidate,
            )

        if isinstance(rule, DecisionRule):
            return DecisionRuleChecker._check_decision_rule(
                rule=rule,
                candidate=candidate,
            )

        raise TypeError(f"Unsupported rule type: {type(rule).__name__}")

    @staticmethod
    def _check_condition(
        rule: Condition,
        candidate: RuleCheckerCandidate,
    ) -> bool:
        feature_value = candidate.get_value(rule.feature_name)

        return rule.operator.apply(
            left_value=float(feature_value),
            right_value=rule.threshold,
        )

    @staticmethod
    def _check_decision_rule(
        rule: DecisionRule,
        candidate: RuleCheckerCandidate,
    ) -> bool:
        return all(
            DecisionRuleChecker.check(
                rule=simple_rule,
                candidate=candidate,
            )
            for simple_rule in rule.conditions
        )


class DecisionRulesFactory:
    @staticmethod
    def build(trained_tree: TrainedDecisionTree) -> list[DecisionRule]:
        classifier = trained_tree.classifier
        tree = classifier.tree_

        feature_names = list(trained_tree.dataset.X.columns)
        class_names = list(classifier.classes_)

        total_train_samples = int(tree.n_node_samples[0])

        rules: list[DecisionRule] = []

        def recurse(
            node_id: int,
            current_rules: list[Condition],
        ) -> None:
            feature_index = tree.feature[node_id]

            if feature_index == _tree.TREE_UNDEFINED:
                class_distribution = tree.value[node_id][0]

                support = int(tree.n_node_samples[node_id])

                predicted_class_index = int(np.argmax(class_distribution))
                predicted_class = class_names[predicted_class_index]

                prediction_probability = float(
                    class_distribution[predicted_class_index]
                )

                score = (
                    support / total_train_samples
                ) * prediction_probability

                rules.append(
                    DecisionRule(
                        predicted_class=predicted_class,
                        prediction_probability=prediction_probability,
                        support=support,
                        score=float(score),
                        conditions=current_rules.copy(),
                    )
                )
                return

            feature_name = feature_names[feature_index]
            threshold = float(tree.threshold[node_id])

            recurse(
                node_id=tree.children_left[node_id],
                current_rules=current_rules
                + [
                    Condition(
                        feature_name=feature_name,
                        threshold=threshold,
                        operator=DecisionOperator.LOWER_EQUAL,
                    )
                ],
            )

            recurse(
                node_id=tree.children_right[node_id],
                current_rules=current_rules
                + [
                    Condition(
                        feature_name=feature_name,
                        threshold=threshold,
                        operator=DecisionOperator.GREATER,
                    )
                ],
            )

        recurse(node_id=0, current_rules=[])

        return rules