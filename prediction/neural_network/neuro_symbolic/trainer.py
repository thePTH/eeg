import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataclasses import dataclass
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from rules.differentiable_rule import DifferentiableDecisionRule
from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsSupervisedLossAggregator,
    MicroLogitsToMacroProbabilityAggregator,
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

    threshold: float = 0.5
    tensorboard_log_dir: str = "tests/neuro_symbolic_eeg"


class NeuroSymbolicDeepEEGTrainer:
    def __init__(self, params: NeuroSymbolicDeepEEGTrainerParameters):
        self.params = params

    def _compute_batch_losses_and_predictions(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        micro_x_raws: torch.Tensor,
        macro_x_feat: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device,
    ):
        micro_x_raws = micro_x_raws.to(device)
        macro_x_feat = macro_x_feat.to(device)
        y_true = y_true.to(device).float()

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

        logic_loss = torch.zeros((), device=device)

        for rule in rules:
            logic_loss = logic_loss + ConditionalViolationLossEngine.compute(
                rule=rule,
                macro_ad_proba=macro_ad_proba,
                x_feat=macro_x_feat,
            )

        lambda_logic = self.params.lambda_logic

        total_loss = (
            (1.0 - lambda_logic) * supervised_loss
            + lambda_logic * logic_loss
        )

        y_pred = (macro_ad_proba >= self.params.threshold).float()

        correct = (y_pred == y_true).sum().item()
        total = y_true.numel()

        return supervised_loss, logic_loss, total_loss, correct, total

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

                supervised_loss, logic_loss, total_loss, correct, total = (
                    self._compute_batch_losses_and_predictions(
                        model=model,
                        rules=rules,
                        micro_x_raws=micro_x_raws,
                        macro_x_feat=macro_x_feat,
                        y_true=y_true,
                        device=device,
                    )
                )

                if train:
                    total_loss.backward()
                    optimizer.step()

                running_supervised_loss += supervised_loss.item()
                running_logic_loss += logic_loss.item()
                running_total_loss += total_loss.item()

                running_correct += correct
                running_total += total

                n_batches = batch_idx + 1

                progress_bar.set_postfix(
                    total_loss=f"{running_total_loss / n_batches:.4f}",
                    accuracy=f"{running_correct / running_total:.4f}",
                )

        n_batches = len(dataloader)

        return {
            "supervised_loss": running_supervised_loss / n_batches,
            "logic_loss": running_logic_loss / n_batches,
            "total_loss": running_total_loss / n_batches,
            "accuracy": running_correct / running_total,
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
                    history[f"val_{key}"].append(value)
                    writer.add_scalar(f"Validation/{key}", value, epoch)

                print(
                    f"Train loss: {train_metrics['total_loss']:.4f} | "
                    f"Train acc: {train_metrics['accuracy']:.4f} | "
                    f"Val loss: {val_metrics['total_loss']:.4f} | "
                    f"Val acc: {val_metrics['accuracy']:.4f}"
                )

            else:
                print(
                    f"Train loss: {train_metrics['total_loss']:.4f} | "
                    f"Train acc: {train_metrics['accuracy']:.4f}"
                )

        writer.close()

        if return_history:
            return model, history

        return model