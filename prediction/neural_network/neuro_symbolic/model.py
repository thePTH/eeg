from rules.differentiable_rule import DifferentiableDecisionRule
from prediction.neural_network.neural_backbone.model import MultiScaleDeepEEGNet
import torch
from prediction.neural_network.helpers import MacroToMicroSegmenter

from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsToMacroProbabilityAggregator,
)

class TrainedNeuroSymbolicNeuralNetwork:
    def __init__(self, neural_backbone_model:MultiScaleDeepEEGNet, logic_rules:DifferentiableDecisionRule):
        self.model = neural_backbone_model
        self.rules = logic_rules

    def __call__(self, macro_x_raw:torch.Tensor):
       

        # 1. Macro EEG -> micro EEG
        micro_x_raws = MacroToMicroSegmenter.split(
            macro_x_raw,
            n_micro_segments=60,
        )

        # 2. Micro EEG -> micro logits
        micro_logits = torch.stack([self.model(micro_x_raw).squeeze(-1) for micro_x_raw in micro_x_raws])


        return MicroLogitsToMacroProbabilityAggregator.compute(
            micro_logits
        )




        





