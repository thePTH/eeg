from __future__ import annotations

import torch


class BinaryMacroProbabilityPrediction:
    """
    A
    Convention :
    ----------
    positive_class_name : classe correspondant à p_ad.
    macro_proba_positive : Tensor [batch_size]
    """

    def __init__(self, macro_proba_positive: torch.Tensor, positive_class_name: str | int="Alzheimer") -> None:
        if macro_proba_positive.ndim != 1:
            raise ValueError(
                "macro_proba_positive must have shape [batch_size]. "
                f"Got {macro_proba_positive.shape}."
            )

        self.macro_proba_positive = macro_proba_positive
        self.positive_class_name = str(positive_class_name)

    def probability(self, class_name: str | int) -> torch.Tensor:
        class_name = str(class_name)

        if class_name == self.positive_class_name:
            return self.macro_proba_positive

        return 1.0 - self.macro_proba_positive