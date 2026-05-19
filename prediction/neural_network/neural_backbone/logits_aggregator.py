from __future__ import annotations

import torch
import torch.nn.functional as F


class MicroLogitsToMacroProbabilityAggregator:
    """
    Agrège des logits micro en une probabilité macro.

    Hypothèses
    ----------
    micro_logits.shape = [S, B]

    avec :
        S = nombre de micro-segments
        B = batch size

    Retour
    ------
    macro_proba.shape = [B]
    """

    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        method: str = "mean_logit",
        
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        micro_logits:
            Tensor de shape [S, B]

        method:
            Méthode d'agrégation parmi :
            - "mean_logit"
            - "mean_probability"
            - "max_probability"
            


        Returns
        -------
        Tensor [B]
        """



        # ==========================================================
        # 1. Mean of logits
        # ==========================================================
        if method == "mean_logit":
            macro_logits = micro_logits.mean(dim=0)
            return torch.sigmoid(macro_logits)

        # ==========================================================
        # 2. Mean of probabilities
        # ==========================================================
        elif method == "mean_probability":
            micro_proba = torch.sigmoid(micro_logits)
            return micro_proba.mean(dim=0)

        # ==========================================================
        # 3. Max probability
        # ==========================================================
        elif method == "max_probability":
            micro_proba = torch.sigmoid(micro_logits)
            return micro_proba.max(dim=0).values



class MicroLogitsSupervisedLossAggregator:
    @staticmethod
    def compute(
        micro_logits: torch.Tensor,
        y_true: torch.Tensor,
        method: str = "macro_bce",
        macro_aggregation_method: str = "mean_logit",
    ) -> torch.Tensor:

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