from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MacroProbabilityAggregationParams:
    method: str = "mean_logit"
    threshold: float = 0.5
    temperature: float = 1.0
    eps: float = 1e-6


@dataclass(frozen=True)
class MicroSupervisedLossAggregationParams:
    method: str = "mean_micro_bce"
    macro_probability_method: str = "mean_logit"
    reduction: str = "mean"
    eps: float = 1e-6


class MicroLogitsToMacroProbabilityAggregator:
    """
    Agrège des micro_logits issus de segments 1s en une macro-probabilité.

    Input
    -----
    micro_logits:
        Tensor de shape [n_micro_segments]
        ou [batch_size, n_micro_segments]

    Output
    ------
    macro_proba_ad:
        Tensor de shape [batch_size]
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        params: MacroProbabilityAggregationParams | None = None,
    ) -> torch.Tensor:
        params = params or MacroProbabilityAggregationParams()

        micro_logits = MicroLogitsToMacroProbabilityAggregator._ensure_2d(
            micro_logits
        )

        micro_probas_ad = torch.sigmoid(micro_logits)

        if params.method == "mean_logit":
            return torch.sigmoid(micro_logits.mean(dim=1))

        if params.method == "mean_probability":
            return micro_probas_ad.mean(dim=1)

        if params.method == "median_probability":
            return micro_probas_ad.median(dim=1).values

        if params.method == "max_probability":
            return micro_probas_ad.max(dim=1).values

        if params.method == "min_probability":
            return micro_probas_ad.min(dim=1).values

        if params.method == "noisy_or":
            micro_probas_ad = torch.clamp(
                micro_probas_ad,
                params.eps,
                1.0 - params.eps,
            )
            return 1.0 - torch.prod(1.0 - micro_probas_ad, dim=1)

        if params.method == "majority_vote":
            return (micro_probas_ad >= params.threshold).float().mean(dim=1)

        if params.method == "soft_majority_vote":
            soft_votes_ad = torch.sigmoid(
                (micro_probas_ad - params.threshold) / params.temperature
            )
            return soft_votes_ad.mean(dim=1)

        if params.method == "confidence_weighted_probability":
            weights = torch.abs(micro_probas_ad - 0.5)
            weights = weights / (
                weights.sum(dim=1, keepdim=True) + params.eps
            )
            return (weights * micro_probas_ad).sum(dim=1)

        if params.method == "attention_probability":
            weights = torch.softmax(
                torch.abs(micro_logits) / params.temperature,
                dim=1,
            )
            return (weights * micro_probas_ad).sum(dim=1)

        raise ValueError(
            f"Unknown macro probability aggregation method: {params.method}"
        )

    @staticmethod
    def _ensure_2d(micro_logits: torch.Tensor) -> torch.Tensor:
        if micro_logits.ndim == 1:
            return micro_logits.unsqueeze(0)

        if micro_logits.ndim != 2:
            raise ValueError(
                "micro_logits must have shape [n_micro_segments] "
                "or [batch_size, n_micro_segments]. "
                f"Got {micro_logits.shape}."
            )

        return micro_logits


class MicroLogitsSupervisedLossAggregator:
    """
    Calcule une loss supervisée à partir des micro_logits et d'un label macro.

    Input
    -----
    micro_logits:
        Tensor de shape [n_micro_segments]
        ou [batch_size, n_micro_segments]

    macro_y_true:
        Tensor scalaire ou Tensor de shape [batch_size]

    Output
    ------
    loss:
        Tensor scalaire
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        macro_y_true: torch.Tensor,
        params: MicroSupervisedLossAggregationParams | None = None,
    ) -> torch.Tensor:
        params = params or MicroSupervisedLossAggregationParams()

        micro_logits = MicroLogitsToMacroProbabilityAggregator._ensure_2d(
            micro_logits
        )

        batch_size = micro_logits.shape[0]

        macro_y_true = MicroLogitsSupervisedLossAggregator._ensure_target_1d(
            macro_y_true=macro_y_true,
            batch_size=batch_size,
            device=micro_logits.device,
        )

        macro_y_true_expanded = macro_y_true.unsqueeze(1).expand_as(
            micro_logits
        )

        if params.method == "mean_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )
            return micro_bce.mean()

        if params.method == "sum_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )
            return micro_bce.sum()

        if params.method == "max_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )
            return micro_bce.max(dim=1).values.mean()

        if params.method == "min_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )
            return micro_bce.min(dim=1).values.mean()

        if params.method == "macro_bce":
            macro_proba_ad = MicroLogitsToMacroProbabilityAggregator.compute(
                micro_logits=micro_logits,
                params=MacroProbabilityAggregationParams(
                    method=params.macro_probability_method,
                    eps=params.eps,
                ),
            )

            macro_proba_ad = torch.clamp(
                macro_proba_ad,
                params.eps,
                1.0 - params.eps,
            )

            return F.binary_cross_entropy(
                macro_proba_ad,
                macro_y_true,
                reduction=params.reduction,
            )

        if params.method == "macro_bce_with_logits":
            if params.macro_probability_method == "mean_logit":
                macro_logit = micro_logits.mean(dim=1)
            else:
                macro_proba_ad = MicroLogitsToMacroProbabilityAggregator.compute(
                    micro_logits=micro_logits,
                    params=MacroProbabilityAggregationParams(
                        method=params.macro_probability_method,
                        eps=params.eps,
                    ),
                )

                macro_logit = MicroLogitsSupervisedLossAggregator._safe_logit(
                    macro_proba_ad,
                    eps=params.eps,
                )

            return F.binary_cross_entropy_with_logits(
                macro_logit,
                macro_y_true,
                reduction=params.reduction,
            )

        if params.method == "confidence_weighted_micro_bce":
            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            micro_probas_ad = torch.sigmoid(micro_logits)

            weights = torch.abs(micro_probas_ad - 0.5)
            weights = weights / (
                weights.sum(dim=1, keepdim=True) + params.eps
            )

            return (weights * micro_bce).sum(dim=1).mean()

        if params.method == "focal_micro_bce":
            gamma = 2.0

            micro_bce = F.binary_cross_entropy_with_logits(
                micro_logits,
                macro_y_true_expanded,
                reduction="none",
            )

            micro_probas_ad = torch.sigmoid(micro_logits)

            p_t = (
                macro_y_true_expanded * micro_probas_ad
                + (1.0 - macro_y_true_expanded)
                * (1.0 - micro_probas_ad)
            )

            focal_weight = (1.0 - p_t) ** gamma

            return (focal_weight * micro_bce).mean()

        raise ValueError(
            f"Unknown micro supervised loss method: {params.method}"
        )

    @staticmethod
    def _ensure_target_1d(
        macro_y_true: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not torch.is_tensor(macro_y_true):
            macro_y_true = torch.tensor(macro_y_true)

        macro_y_true = macro_y_true.float().to(device)

        if macro_y_true.ndim == 0:
            macro_y_true = macro_y_true.unsqueeze(0)

        if macro_y_true.ndim != 1:
            raise ValueError(
                "macro_y_true must have shape [] or [batch_size]. "
                f"Got {macro_y_true.shape}."
            )

        if macro_y_true.shape[0] != batch_size:
            raise ValueError(
                f"macro_y_true has shape {macro_y_true.shape}, "
                f"but batch_size is {batch_size}."
            )

        return macro_y_true

    @staticmethod
    def _safe_logit(
        proba: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        proba = torch.clamp(proba, eps, 1.0 - eps)
        return torch.logit(proba)