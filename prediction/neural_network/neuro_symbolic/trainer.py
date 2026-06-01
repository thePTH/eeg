import warnings

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import balanced_accuracy_score

from rules.differentiable_rule import DifferentiableDecisionRule
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsSupervisedLossAggregator,
    MicroLogitsToMacroProbabilityAggregator,
)
from prediction.neural_network.neuro_symbolic.logic_loss import (
    ConditionalViolationLossEngine,
)


# =============================================================================
# Disable sklearn warning raised when temporary subsets of predictions
# contain classes not yet present in y_true.
# This can happen during progressive metric accumulation.
# =============================================================================

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
)


@dataclass
class NeuroSymbolicDeepEEGTrainerParameters:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_logic: float = 0.5

    macro_aggregation_method: str = "mean_logit"
    supervised_loss_compute_method: str = "micro_bce"

    threshold: float = 0.5
    tensorboard_log_dir: str = "tests/neuro_symbolic_eeg"


class NeuroSymbolicDeepEEGTrainer:
    def __init__(self, params: NeuroSymbolicDeepEEGTrainerParameters):
        self.params = params

    def _forward_micro_segments(
        self,
        model: nn.Module,
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes logits for all micro-segments.

        Expected input shape:
            micro_x_raws: [batch, n_micro_segments, channels, samples]

        Returned shape:
            micro_logits: [n_micro_segments, batch]
        """

        if micro_x_raws.ndim != 4:
            raise ValueError(
                "Expected micro_x_raws with shape "
                "[batch, n_micro_segments, channels, samples]. "
                f"Got {micro_x_raws.shape}."
            )

        micro_x_raws = micro_x_raws.permute(1, 0, 2, 3).contiguous()

        micro_logits = torch.stack(
            [
                model(micro_x_raw).squeeze(-1)
                for micro_x_raw in micro_x_raws
            ],
            dim=0,
        )

        return micro_logits

    def _compute_logic_loss(
        self,
        rules: list[DifferentiableDecisionRule],
        macro_ad_proba: torch.Tensor,
        macro_x_feat: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Computes the total differentiable logic loss over all rules.
        """

        logic_loss = torch.zeros((), device=device)

        for rule in rules:
            logic_loss = logic_loss + ConditionalViolationLossEngine.compute(
                rule=rule,
                macro_ad_proba=macro_ad_proba,
                x_feat=macro_x_feat,
            )

        return logic_loss

    def _compute_batch_outputs(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        micro_x_raws: torch.Tensor,
        macro_x_feat: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device,
    ) -> dict[str, torch.Tensor | float | int]:
        """
        Computes losses, probabilities and predictions for one batch.

        This function performs a single forward pass and returns everything
        needed for training, metrics, TensorBoard and console logs.
        """

        micro_x_raws = micro_x_raws.to(device)
        macro_x_feat = macro_x_feat.to(device)
        y_true = y_true.to(device).float()

        micro_logits = self._forward_micro_segments(
            model=model,
            micro_x_raws=micro_x_raws,
        )

        supervised_loss = MicroLogitsSupervisedLossAggregator.compute(
            micro_logits,
            y_true,
            method=self.params.supervised_loss_compute_method,
            macro_aggregation_method=self.params.macro_aggregation_method,
        )

        macro_ad_proba = MicroLogitsToMacroProbabilityAggregator.compute(
            micro_logits,
            method=self.params.macro_aggregation_method,
        )

        logic_loss = self._compute_logic_loss(
            rules=rules,
            macro_ad_proba=macro_ad_proba,
            macro_x_feat=macro_x_feat,
            device=device,
        )

        total_loss = (
            (1.0 - self.params.lambda_logic) * supervised_loss
            + self.params.lambda_logic * logic_loss
        )

        y_pred = (macro_ad_proba >= self.params.threshold).float()

        correct = (y_pred == y_true).sum().item()
        total = y_true.numel()

        return {
            "supervised_loss": supervised_loss,
            "logic_loss": logic_loss,
            "total_loss": total_loss,
            "y_true": y_true.detach().cpu(),
            "y_pred": y_pred.detach().cpu(),
            "correct": correct,
            "total": total,
        }

    def _run_epoch(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        dataloader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None = None,
        train: bool = True,
    ) -> dict[str, float]:

        if train:
            model.train()
        else:
            model.eval()

        running_supervised_loss = 0.0
        running_logic_loss = 0.0
        running_total_loss = 0.0

        running_correct = 0
        running_total = 0

        all_y_true = []
        all_y_pred = []

        context = torch.enable_grad() if train else torch.no_grad()

        with context:
            progress_bar = tqdm(
                dataloader,
                desc="Train" if train else "Validation",
                leave=False,
            )

            for batch_idx, (micro_x_raws, macro_x_feat, y_true) in enumerate(progress_bar):

                if train:
                    optimizer.zero_grad()

                batch_outputs = self._compute_batch_outputs(
                    model=model,
                    rules=rules,
                    micro_x_raws=micro_x_raws,
                    macro_x_feat=macro_x_feat,
                    y_true=y_true,
                    device=device,
                )

                total_loss = batch_outputs["total_loss"]

                if train:
                    total_loss.backward()
                    optimizer.step()

                running_supervised_loss += batch_outputs["supervised_loss"].item()
                running_logic_loss += batch_outputs["logic_loss"].item()
                running_total_loss += batch_outputs["total_loss"].item()

                running_correct += batch_outputs["correct"]
                running_total += batch_outputs["total"]

                all_y_true.extend(batch_outputs["y_true"].numpy().tolist())
                all_y_pred.extend(batch_outputs["y_pred"].numpy().tolist())

                n_batches = batch_idx + 1

                standard_accuracy = running_correct / running_total

                current_balanced_accuracy = balanced_accuracy_score(
                    all_y_true,
                    all_y_pred,
                )

                progress_bar.set_postfix(
                    total_loss=f"{running_total_loss / n_batches:.4f}",
                    acc=f"{standard_accuracy:.4f}",
                    balanced_acc=f"{current_balanced_accuracy:.4f}",
                )

        n_batches = len(dataloader)

        standard_accuracy = running_correct / running_total

        balanced_accuracy = balanced_accuracy_score(
            all_y_true,
            all_y_pred,
        )

        return {
            "supervised_loss": running_supervised_loss / n_batches,
            "logic_loss": running_logic_loss / n_batches,
            "total_loss": running_total_loss / n_batches,

            # Important:
            # For history and TensorBoard, the key remains "accuracy",
            # but the stored value is now the balanced accuracy.
            "accuracy": balanced_accuracy,

            # Console-only metric.
            "standard_accuracy": standard_accuracy,
        }

    def train(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        return_history: bool = False,
    ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        writer = SummaryWriter(log_dir=self.params.tensorboard_log_dir)

        history = {
            "train_supervised_loss": [],
            "train_logic_loss": [],
            "train_total_loss": [],
            "train_accuracy": [],
            "val_supervised_loss": [],
            "val_logic_loss": [],
            "val_total_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(self.params.epochs):

            print(f"\nEpoch {epoch + 1}/{self.params.epochs}")

            train_metrics = self._run_epoch(
                model=model,
                rules=rules,
                dataloader=train_dataloader,
                device=device,
                optimizer=optimizer,
                train=True,
            )

            for key, value in train_metrics.items():

                if key == "standard_accuracy":
                    continue

                history[f"train_{key}"].append(value)
                writer.add_scalar(f"Train/{key}", value, epoch)

            if val_dataloader is not None:

                val_metrics = self._run_epoch(
                    model=model,
                    rules=rules,
                    dataloader=val_dataloader,
                    device=device,
                    optimizer=None,
                    train=False,
                )

                for key, value in val_metrics.items():

                    if key == "standard_accuracy":
                        continue

                    history[f"val_{key}"].append(value)
                    writer.add_scalar(f"Validation/{key}", value, epoch)

                print(
                    f"Train loss: {train_metrics['total_loss']:.4f} | "
                    f"Train acc: {train_metrics['standard_accuracy']:.4f} | "
                    f"Train balanced acc: {train_metrics['accuracy']:.4f} | "
                    f"Val loss: {val_metrics['total_loss']:.4f} | "
                    f"Val acc: {val_metrics['standard_accuracy']:.4f} | "
                    f"Val balanced acc: {val_metrics['accuracy']:.4f}"
                )

            else:

                print(
                    f"Train loss: {train_metrics['total_loss']:.4f} | "
                    f"Train acc: {train_metrics['standard_accuracy']:.4f} | "
                    f"Train balanced acc: {train_metrics['accuracy']:.4f}"
                )

            writer.flush()

        writer.close()

        if return_history:
            return model, history

        return model