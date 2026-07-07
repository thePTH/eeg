import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from prediction.neural_network.neural_backbone.logits_aggregator import (
    MicroLogitsSupervisedLossAggregator,
    MicroLogitsToMacroProbabilityAggregator,
)
from prediction.neural_network.neuro_symbolic.logic_loss import (
    ConditionalViolationLossEngine,
)
from rules.differentiable_rule import DifferentiableDecisionRule


warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
)

warnings.filterwarnings(
    "ignore",
    message="A single label was found in 'y_true' and 'y_pred'.*",
)


@dataclass
class NeuroSymbolicDeepEEGTrainerParameters:
    """Configuration parameters for neuro-symbolic DeepEEG training."""

    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_logic: float = 0.5

    macro_aggregation_method: str = "mean_logit"
    supervised_loss_compute_method: str = "micro_bce"

    threshold: float = 0.5
    tensorboard_log_dir: str = "tests/neuro_symbolic_eeg"

    loss_scale_alpha: float = 0.99
    loss_scale_eps: float = 1e-8


class NeuroSymbolicDeepEEGTrainer:
    """Trainer for a DeepEEG model regularized with differentiable logical rules."""

    def __init__(
        self,
        params: NeuroSymbolicDeepEEGTrainerParameters,
    ):
        self.params = params
        self.supervised_loss_scale = None
        self.logic_loss_scale = None

    def _forward_micro_segments(
        self,
        model: nn.Module,
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:
        """Forward all micro-segments through the neural model."""
        if micro_x_raws.ndim != 4:
            raise ValueError(
                "Expected micro_x_raws with shape "
                "[batch, n_micro_segments, channels, samples]. "
                f"Got {micro_x_raws.shape}."
            )

        micro_x_raws = micro_x_raws.permute(
            1,
            0,
            2,
            3,
        ).contiguous()

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
        """Compute the summed conditional-violation loss over all rules."""
        logic_loss = torch.zeros(
            (),
            device=device,
        )

        for rule in rules:
            logic_loss = (
                logic_loss
                + ConditionalViolationLossEngine.compute(
                    rule=rule,
                    macro_ad_proba=macro_ad_proba,
                    x_feat=macro_x_feat,
                )
            )

        return logic_loss

    def _update_loss_scales(
        self,
        supervised_loss: torch.Tensor,
        logic_loss: torch.Tensor,
    ) -> None:
        """Update moving-average loss scales used for logic-loss normalization."""
        with torch.no_grad():

            supervised_value = supervised_loss.detach()
            logic_value = logic_loss.detach()

            if self.supervised_loss_scale is None:
                self.supervised_loss_scale = supervised_value

            else:
                self.supervised_loss_scale = (
                    self.params.loss_scale_alpha
                    * self.supervised_loss_scale
                    + (1.0 - self.params.loss_scale_alpha)
                    * supervised_value
                )

            if self.logic_loss_scale is None:
                self.logic_loss_scale = logic_value

            else:
                self.logic_loss_scale = (
                    self.params.loss_scale_alpha
                    * self.logic_loss_scale
                    + (1.0 - self.params.loss_scale_alpha)
                    * logic_value
                )

    def _normalize_logic_loss(
        self,
        supervised_loss: torch.Tensor,
        logic_loss: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        """Rescale the logic loss to keep it comparable to the supervised loss."""
        if train:
            self._update_loss_scales(
                supervised_loss=supervised_loss,
                logic_loss=logic_loss,
            )

        if self.supervised_loss_scale is None:
            self.supervised_loss_scale = supervised_loss.detach()

        if self.logic_loss_scale is None:
            self.logic_loss_scale = logic_loss.detach()

        normalized_logic_loss = (
            logic_loss
            * self.supervised_loss_scale
            / (
                self.logic_loss_scale
                + self.params.loss_scale_eps
            )
        )

        return normalized_logic_loss

    def _compute_batch_outputs(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        micro_x_raws: torch.Tensor,
        macro_x_feat: torch.Tensor,
        y_true: torch.Tensor,
        device: torch.device,
        train: bool,
    ) -> dict[str, torch.Tensor | float | int]:
        """Compute losses, predictions, and metrics for one batch."""
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
            macro_aggregation_method=(
                self.params.macro_aggregation_method
            ),
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

        normalized_logic_loss = self._normalize_logic_loss(
            supervised_loss=supervised_loss,
            logic_loss=logic_loss,
            train=train,
        )

        total_loss = (
            (1.0 - self.params.lambda_logic)
            * supervised_loss
            + self.params.lambda_logic
            * normalized_logic_loss
        )

        y_pred = (
            macro_ad_proba >= self.params.threshold
        ).float()

        correct = (
            y_pred == y_true
        ).sum().item()

        total = y_true.numel()

        return {
            "supervised_loss": supervised_loss,
            "logic_loss": logic_loss,
            "normalized_logic_loss": normalized_logic_loss,
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
        """Run one training or validation epoch."""
        if train:
            model.train()
        else:
            model.eval()

        running_supervised_loss = 0.0
        running_logic_loss = 0.0
        running_normalized_logic_loss = 0.0
        running_total_loss = 0.0

        running_correct = 0
        running_total = 0

        all_y_true = []
        all_y_pred = []

        context = (
            torch.enable_grad()
            if train
            else torch.no_grad()
        )

        with context:
            progress_bar = tqdm(
                dataloader,
                desc="Train" if train else "Validation",
                leave=False,
            )

            for batch_idx, (
                micro_x_raws,
                macro_x_feat,
                y_true,
            ) in enumerate(progress_bar):

                if train:
                    optimizer.zero_grad()

                batch_outputs = self._compute_batch_outputs(
                    model=model,
                    rules=rules,
                    micro_x_raws=micro_x_raws,
                    macro_x_feat=macro_x_feat,
                    y_true=y_true,
                    device=device,
                    train=train,
                )

                total_loss = batch_outputs["total_loss"]

                if train:
                    total_loss.backward()
                    optimizer.step()

                running_supervised_loss += (
                    batch_outputs["supervised_loss"].item()
                )

                running_logic_loss += (
                    batch_outputs["logic_loss"].item()
                )

                running_normalized_logic_loss += (
                    batch_outputs["normalized_logic_loss"].item()
                )

                running_total_loss += (
                    batch_outputs["total_loss"].item()
                )

                running_correct += batch_outputs["correct"]
                running_total += batch_outputs["total"]

                all_y_true.extend(
                    batch_outputs["y_true"].numpy().tolist()
                )

                all_y_pred.extend(
                    batch_outputs["y_pred"].numpy().tolist()
                )

                n_batches = batch_idx + 1

                standard_accuracy = (
                    running_correct / running_total
                )

                progress_bar.set_postfix(
                    total_loss=(
                        f"{running_total_loss / n_batches:.4f}"
                    ),
                    sup_loss=(
                        f"{running_supervised_loss / n_batches:.4f}"
                    ),
                    logic_loss=(
                        f"{running_logic_loss / n_batches:.4f}"
                    ),
                    norm_logic=(
                        f"{running_normalized_logic_loss / n_batches:.4f}"
                    ),
                    acc=f"{standard_accuracy:.4f}",
                )

        n_batches = len(dataloader)

        standard_accuracy = (
            running_correct / running_total
        )

        balanced_accuracy = balanced_accuracy_score(
            all_y_true,
            all_y_pred,
        )

        return {
            "supervised_loss": (
                running_supervised_loss / n_batches
            ),
            "logic_loss": (
                running_logic_loss / n_batches
            ),
            "normalized_logic_loss": (
                running_normalized_logic_loss / n_batches
            ),
            "total_loss": (
                running_total_loss / n_batches
            ),
            "accuracy": balanced_accuracy,
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
        """Train the model and optionally return the metric history."""
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = model.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.params.lr,
            weight_decay=self.params.weight_decay,
        )

        writer = SummaryWriter(
            log_dir=self.params.tensorboard_log_dir
        )

        history = {
            "train_supervised_loss": [],
            "train_logic_loss": [],
            "train_normalized_logic_loss": [],
            "train_total_loss": [],
            "train_accuracy": [],
            "val_supervised_loss": [],
            "val_logic_loss": [],
            "val_normalized_logic_loss": [],
            "val_total_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(self.params.epochs):

            print(
                f"\nEpoch {epoch + 1}/{self.params.epochs}"
            )

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

                writer.add_scalar(
                    f"Train/{key}",
                    value,
                    epoch,
                )

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

                    writer.add_scalar(
                        f"Validation/{key}",
                        value,
                        epoch,
                    )

                print(
                    f"Train loss: "
                    f"{train_metrics['total_loss']:.4f} | "
                    f"Train sup: "
                    f"{train_metrics['supervised_loss']:.4f} | "
                    f"Train logic: "
                    f"{train_metrics['logic_loss']:.4f} | "
                    f"Train norm logic: "
                    f"{train_metrics['normalized_logic_loss']:.4f} | "
                    f"Train acc: "
                    f"{train_metrics['standard_accuracy']:.4f} | "
                    f"Train balanced acc: "
                    f"{train_metrics['accuracy']:.4f} | "
                    f"Val loss: "
                    f"{val_metrics['total_loss']:.4f} | "
                    f"Val acc: "
                    f"{val_metrics['standard_accuracy']:.4f} | "
                    f"Val balanced acc: "
                    f"{val_metrics['accuracy']:.4f}"
                )

            else:

                print(
                    f"Train loss: "
                    f"{train_metrics['total_loss']:.4f} | "
                    f"Train sup: "
                    f"{train_metrics['supervised_loss']:.4f} | "
                    f"Train logic: "
                    f"{train_metrics['logic_loss']:.4f} | "
                    f"Train norm logic: "
                    f"{train_metrics['normalized_logic_loss']:.4f} | "
                    f"Train acc: "
                    f"{train_metrics['standard_accuracy']:.4f} | "
                    f"Train balanced acc: "
                    f"{train_metrics['accuracy']:.4f}"
                )

            writer.flush()

        writer.close()

        if return_history:
            return model, history

        return model

    def evaluate(
        self,
        model: nn.Module,
        rules: list[DifferentiableDecisionRule],
        dataloader: DataLoader,
    ) -> dict[str, float]:
        """Evaluate the trained model on a dataloader."""
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model = model.to(device)
        model.eval()

        all_y_true = []
        all_y_pred = []

        running_total_loss = 0.0
        running_normalized_logic_loss = 0.0

        with torch.no_grad():

            for (
                micro_x_raws,
                macro_x_feat,
                y_true,
            ) in tqdm(
                dataloader,
                desc="Test",
                leave=False,
            ):

                batch_outputs = self._compute_batch_outputs(
                    model=model,
                    rules=rules,
                    micro_x_raws=micro_x_raws,
                    macro_x_feat=macro_x_feat,
                    y_true=y_true,
                    device=device,
                    train=False,
                )

                running_total_loss += (
                    batch_outputs["total_loss"].item()
                )

                running_normalized_logic_loss += (
                    batch_outputs[
                        "normalized_logic_loss"
                    ].item()
                )

                all_y_true.extend(
                    batch_outputs["y_true"].numpy().tolist()
                )

                all_y_pred.extend(
                    batch_outputs["y_pred"].numpy().tolist()
                )

        return {
            "test_total_loss": (
                running_total_loss / len(dataloader)
            ),
            "test_normalized_logic_loss": (
                running_normalized_logic_loss / len(dataloader)
            ),
            "test_balanced_accuracy": balanced_accuracy_score(
                all_y_true,
                all_y_pred,
            ),
            "test_f1_score": f1_score(
                all_y_true,
                all_y_pred,
            ),
        }