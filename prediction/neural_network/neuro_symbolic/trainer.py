import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataclasses import dataclass

from tqdm.auto import tqdm

from rules.differentiable_rule import DifferentiableDecisionRule
from prediction.neural_network.helpers import MacroToMicroSegmenter
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsSupervisedLossAggregator,
    MicroLogitsToMacroProbabilityAggregator
)
from prediction.neural_network.neuro_symbolic.logic_loss import (
    ConditionalViolationLossEngine,
)


@dataclass
class NeuroSymbolicDeepEEGTrainerParameters:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_logic: float = 0.5

    macro_aggregation_method: str = "mean_logit"
    supervised_loss_compute_method: str = "micro_bce"


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

        # ==========================================================
        # Training loop
        # ==========================================================
        for epoch in range(self.params.epochs):

            # Running statistics for clean tqdm display
            running_supervised_loss = 0.0
            running_logic_loss = 0.0
            running_total_loss = 0.0

            # Progress bar ONLY on batches
            progress_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{self.params.epochs}",
                leave=True,
            )

            for batch_idx, (macro_x_raw, macro_x_feat, y_true) in enumerate(progress_bar):

                # ==================================================
                # Device
                # ==================================================
                macro_x_raw = macro_x_raw.to(device)
                macro_x_feat = macro_x_feat.to(device)
                y_true = y_true.to(device).float()

                optimizer.zero_grad()

                # ==================================================
                # 1. Macro EEG -> micro EEG
                # Shape:
                # [n_micro_segments, batch, channels, samples]
                # ==================================================
                micro_x_raws = MacroToMicroSegmenter.split(
                    macro_x_raw,
                    n_micro_segments=60,
                )

                # ==================================================
                # 2. Micro EEG -> micro logits
                # Shape:
                # [n_micro_segments, batch]
                # ==================================================
                micro_logits = torch.stack(
                    [
                        model(micro_x_raw).squeeze(-1)
                        for micro_x_raw in micro_x_raws
                    ]
                )

                # ==================================================
                # 3. Supervised loss
                # ==================================================
                supervised_loss = (
                    MicroLogitsSupervisedLossAggregator.compute(
                        micro_logits,
                        y_true,
                        method=self.params.supervised_loss_compute_method,
                        macro_aggregation_method=self.params.macro_aggregation_method,
                    )
                )

                # ==================================================
                # 4. Macro probability aggregation
                # ==================================================
                macro_ad_proba = (
                    MicroLogitsToMacroProbabilityAggregator.compute(
                        micro_logits,
                        method=self.params.macro_aggregation_method,
                    )
                )

                # ==================================================
                # 5. Logic loss
                # ==================================================
                logic_loss = torch.zeros((), device=device)

                for rule in rules:
                    logic_loss = logic_loss + (
                        ConditionalViolationLossEngine.compute(
                            rule=rule,
                            macro_ad_proba=macro_ad_proba,
                            x_feat=macro_x_feat,
                        )
                    )

                # ==================================================
                # 6. Total loss
                # ==================================================
                lambda_logic = self.params.lambda_logic

                total_loss = (
                    (1 - lambda_logic) * supervised_loss
                    + lambda_logic * logic_loss
                )

                # ==================================================
                # Backpropagation
                # ==================================================
                total_loss.backward()
                optimizer.step()

                # ==================================================
                # Running metrics
                # ==================================================
                running_supervised_loss += supervised_loss.item()
                running_logic_loss += logic_loss.item()
                running_total_loss += total_loss.item()

                n_batches = batch_idx + 1

                # ==================================================
                # Update tqdm display
                # ==================================================
                progress_bar.set_postfix(
                    supervised_loss=f"{running_supervised_loss / n_batches:.4f}",
                    logic_loss=f"{running_logic_loss / n_batches:.4f}",
                    total_loss=f"{running_total_loss / n_batches:.4f}",
                )

        return model