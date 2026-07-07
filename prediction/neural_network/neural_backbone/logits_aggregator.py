from __future__ import annotations

import torch
import torch.nn.functional as F


class MicroLogitsToMacroProbabilityAggregator:
    """
    Aggregate micro-level logits into macro-level probabilities.

    Assumptions
    -----------
    micro_logits
        Tensor with shape ``[S, B]``, where:
        - ``S`` is the number of micro-segments,
        - ``B`` is the batch size.

    Returns
    -------
    torch.Tensor
        Macro-level probability tensor with shape ``[B]``.
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        method: str = "mean_logit",
    ) -> torch.Tensor:
        """
        Aggregate micro logits.

        Parameters
        ----------
        micro_logits
            Tensor with shape ``[S, B]``.
        method
            Aggregation method. Supported values are:
            - ``"mean_logit"``
            - ``"mean_probability"``
            - ``"max_probability"``

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[B]``.
        """
        if method == "mean_logit":
            macro_logits = micro_logits.mean(dim=0)
            return torch.sigmoid(macro_logits)

        elif method == "mean_probability":
            micro_proba = torch.sigmoid(micro_logits)
            return micro_proba.mean(dim=0)

        elif method == "max_probability":
            micro_proba = torch.sigmoid(micro_logits)
            return micro_proba.max(dim=0).values


class MicroLogitsSupervisedLossAggregator:
    """Aggregate supervised losses from micro-level logits."""

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        y_true: torch.Tensor,
        method: str = "macro_bce",
        macro_aggregation_method: str = "mean_logit",
    ) -> torch.Tensor:
        """
        Compute supervised binary classification loss from micro logits.

        Parameters
        ----------
        micro_logits
            Tensor with shape ``[S, B]``.
        y_true
            Ground-truth macro labels with shape ``[B]``.
        method
            Loss aggregation method. Supported values are:
            - ``"micro_bce"``
            - ``"macro_bce"``
        macro_aggregation_method
            Macro aggregation strategy used when ``method="macro_bce"`` and
            probabilities must be computed before applying BCE.

        Returns
        -------
        torch.Tensor
            Scalar supervised loss.
        """
        if micro_logits.ndim != 2:
            raise ValueError(
                f"micro_logits must have shape [S, B]. Got {micro_logits.shape}."
            )

        y_true = y_true.float().view(-1)

        S, B = micro_logits.shape

        if y_true.shape[0] != B:
            raise ValueError(
                f"Batch mismatch: micro_logits has B={B}, y_true has {y_true.shape[0]}."
            )

        if method == "micro_bce":
            logits_micro = micro_logits.reshape(S * B)

            y_micro = (
                y_true
                .unsqueeze(0)
                .expand(S, B)
                .reshape(S * B)
            )

            return F.binary_cross_entropy_with_logits(
                logits_micro,
                y_micro,
            )

        elif method == "macro_bce":
            if macro_aggregation_method == "mean_logit":
                macro_logits = micro_logits.mean(dim=0)

                return F.binary_cross_entropy_with_logits(
                    macro_logits,
                    y_true,
                )

            else:
                macro_proba = MicroLogitsToMacroProbabilityAggregator.compute(
                    micro_logits=micro_logits,
                    method=macro_aggregation_method,
                )

                return F.binary_cross_entropy(
                    macro_proba.clamp(1e-6, 1 - 1e-6),
                    y_true,
                )

        else:
            raise ValueError(f"Unknown supervised loss method: {method}")