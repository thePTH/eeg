import torch

from prediction.neural_network.result import BinaryMacroProbabilityPrediction
from rules.differentiable_rule import (
    DifferentiableDecisionRule,
    DifferentiableRuleCandidateFactory,
    TruthDegreeEngine,
)

EPS = 1e-5


class ConditionalViolationLossEngine:
    """Computes the weighted conditional violation loss of a differentiable rule."""

    @staticmethod
    def compute(
        rule: DifferentiableDecisionRule,
        macro_ad_proba: torch.Tensor,
        x_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the expected violation of a rule over a batch.

        The loss is defined as:

            score(rule) * E[1 - P(predicted_class) | rule is satisfied]

        where the expectation is weighted by the differentiable truth degree
        of the rule.
        """
        candidate = DifferentiableRuleCandidateFactory.from_tensor(x_feat)

        prediction = BinaryMacroProbabilityPrediction(macro_ad_proba)

        truth_degree = TruthDegreeEngine.compute(
            rule,
            candidate,
        )

        violation = (
            1.0
            - prediction.probability(rule.predicted_class)
        )

        expected_violation = (
            (truth_degree * violation).sum()
            / (truth_degree.sum() + EPS)
        )

        return rule.score * expected_violation