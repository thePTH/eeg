import torch

from prediction.neural_network.helpers import MacroToMicroSegmenter
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsToMacroProbabilityAggregator,
)
from prediction.neural_network.neural_backbone.model import MultiScaleDeepEEGNet
from rules.differentiable_rule import DifferentiableDecisionRule


class TrainedNeuroSymbolicNeuralNetwork:
    """Wrapper around a trained neural backbone and its differentiable logic rules."""

    def __init__(
        self,
        neural_backbone_model: MultiScaleDeepEEGNet,
        logic_rules: DifferentiableDecisionRule,
    ):
        self.model = neural_backbone_model
        self.rules = logic_rules

    def __call__(self, macro_x_raw: torch.Tensor):
        """Predict macro-level probability from a macro EEG tensor."""
        micro_x_raws = MacroToMicroSegmenter.split(
            macro_x_raw,
            n_micro_segments=60,
        )

        micro_logits = torch.stack(
            [
                self.model(micro_x_raw).squeeze(-1)
                for micro_x_raw in micro_x_raws
            ]
        )

        return MicroLogitsToMacroProbabilityAggregator.compute(
            micro_logits
        )