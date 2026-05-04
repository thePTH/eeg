from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np

class DecisionOperator(str, Enum):
    GREATER = ">"
    LOWER = "<"
    GREATER_EQUAL = ">="
    LOWER_EQUAL = "<="

    def apply(self, left_value: float, right_value: float) -> bool:
        """Applique l'opérateur entre deux valeurs numériques."""
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



@dataclass(frozen=True)
class SimpleDecisionRule:
    feature_name: str
    threshold: float
    operator: DecisionOperator
    
    def evaluate(self, feature_value:float) -> bool:
        """
        Évalue la règle sur un échantillon.
        """
        

        condition_is_true = self.operator.apply(
            left_value=feature_value,
            right_value=self.threshold,
        )

        return condition_is_true

    def __str__(self) -> str:
        return f"{self.feature_name} {self.operator.value} {self.threshold}"
    


@dataclass
class DecisionRule:
    predicted_class: str
    prediction_probability: float  # = purity
    support: int
    score: float
    simple_rules: list[SimpleDecisionRule]

    def evaluate(self, candidate: pd.DataFrame) -> bool:
        row = candidate.iloc[0]

        for simple_rule in self.simple_rules:
            feature_value = row[simple_rule.feature_name]

            if not simple_rule.evaluate(feature_value):
                return False

        return True

    def __str__(self) -> str:
        conds = "\n  AND ".join(str(rule) for rule in self.simple_rules)

        return (
            f"{conds}\n"
            f"=> class={self.predicted_class} "
            f"(n={self.support}, p={self.prediction_probability:.3f}, "
            f"score={self.score:.3f})"
        )

    def __repr__(self) -> str:
        return self.__str__()
    




from prediction.decision_tree.base import TrainedDecisionTree
from sklearn.tree import _tree


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
            current_rules: list[SimpleDecisionRule],
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
                        simple_rules=current_rules.copy(),
                    )
                )
                return

            feature_name = feature_names[feature_index]
            threshold = float(tree.threshold[node_id])

            recurse(
                tree.children_left[node_id],
                current_rules + [
                    SimpleDecisionRule(
                        feature_name=feature_name,
                        threshold=threshold,
                        operator=DecisionOperator.LOWER_EQUAL,
                    )
                ],
            )

            recurse(
                tree.children_right[node_id],
                current_rules + [
                    SimpleDecisionRule(
                        feature_name=feature_name,
                        threshold=threshold,
                        operator=DecisionOperator.GREATER,
                    )
                ],
            )

        recurse(node_id=0, current_rules=[])

        return rules