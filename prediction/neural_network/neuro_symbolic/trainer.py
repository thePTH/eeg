import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass, field

from rules.differentiable_rule import DifferentiableDecisionRule
from prediction.neural_network.helpers import MacroToMicroSegmenter
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsSupervisedLossAggregator,
    MicroLogitsToMacroProbabilityAggregator
)
from prediction.neural_network.neuro_symbolic.logic_loss import (
    ConditionalViolationLossEngine,
)

from prediction.neural_network.neuro_symbolic.model import TrainedNeuroSymbolicNeuralNetwork

@dataclass
class NeuroSymbolicDeepEEGTrainerParameters:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_logic: float = 0.5
    macro_aggregation_method: str = "mean_logit"
    supervised_loss_compute_method : str = "micro_bce"


class NeuroSymbolicDeepEEGTrainer:
    def __init__(self, params: NeuroSymbolicDeepEEGTrainerParameters):
        self.params = params

    def train(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        dataloader: DataLoader,
    ) -> nn.Module:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        model.train()

        for epoch in range(self.params.epochs):
            for macro_x_raw, macro_x_feat, y_true in dataloader:
                macro_x_raw = macro_x_raw.to(device)
                macro_x_feat = macro_x_feat.to(device)
                y_true = y_true.to(device).float()

                optimizer.zero_grad()

                # 1. Macro EEG -> micro EEG
                micro_x_raws = MacroToMicroSegmenter.split( #[60, 8, 19, 500]
                    macro_x_raw,
                    n_micro_segments=60,
                )

                # 2. Micro EEG -> micro logits
                micro_logits = torch.stack([model(micro_x_raw).squeeze(-1) for micro_x_raw in micro_x_raws])

                # 3. Micro logits -> supervised loss
                supervised_loss = MicroLogitsSupervisedLossAggregator.compute(
                    micro_logits,
                    y_true,
                    method=self.params.supervised_loss_compute_method,
                    macro_aggregation_method=self.params.macro_aggregation_method
                )
                #print(supervised_loss)

                # 4. Micro logits -> macro probability
                macro_ad_proba = MicroLogitsToMacroProbabilityAggregator.compute(
                    micro_logits,
                    method=self.params.macro_aggregation_method
                )
                #print(macro_ad_proba)

                # 5. Logic loss
                logic_loss = torch.zeros((), device=device)

                for rule in rules:
                    logic_loss = logic_loss + ConditionalViolationLossEngine.compute(
                        rule=rule,
                        macro_ad_proba=macro_ad_proba,
                        x_feat=macro_x_feat,
                    )

                # 6. Total loss
                lambda_logic = self.params.lambda_logic
                total_loss = (1 - lambda_logic) *  supervised_loss + lambda_logic * logic_loss

                total_loss.backward()
                optimizer.step()
                
            print(
                f"Epoch {epoch + 1} | "
                f"supervised_loss: {supervised_loss.item():.4f} | "
                f"logic_loss: {logic_loss.item():.4f} | "
                f"total_loss: {total_loss.item():.4f}"
            )



        return model