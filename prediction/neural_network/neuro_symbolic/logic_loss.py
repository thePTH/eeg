from rules.differentiable_rule import DifferentiableDecisionRule, TruthDegreeEngine, DifferentiableRuleCandidateFactory
from prediction.neural_network.result import BinaryMacroProbabilityPrediction
import torch

eps = 10e-6

class ConditionalViolationLossEngine:
    @staticmethod
    def compute(rule: DifferentiableDecisionRule, macro_ad_proba:torch.Tensor, x_feat: torch.Tensor) -> torch.Tensor:
    
        candidate = DifferentiableRuleCandidateFactory.from_tensor(x_feat)
        neural_network_predicrion = BinaryMacroProbabilityPrediction(macro_ad_proba)
        truth_degree = TruthDegreeEngine.compute(rule, candidate)

        violation = 1 - neural_network_predicrion.probability(rule.predicted_class)
        expected_violation = (truth_degree * violation).sum() / (truth_degree.sum() + eps)

        return rule.score * expected_violation
    



