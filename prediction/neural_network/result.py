from __future__ import annotations

import torch


class BinaryMacroProbabilityPrediction:
    """
    Binary probability prediction container.

    Convention
    ----------
    - ``positive_class_name`` identifies the positive class.
    - ``macro_proba_positive`` is a tensor of shape ``[batch_size]``
      containing the predicted probability of the positive class.
    """

    def __init__(
        self,
        macro_proba_positive: torch.Tensor,
        positive_class_name: str | int = "Alzheimer",
    ) -> None:
        """
        Parameters
        ----------
        macro_proba_positive:
            Predicted probability of the positive class for each sample.
            Expected shape: ``[batch_size]``.
        positive_class_name:
            Name (or identifier) of the positive class.
        """
        if macro_proba_positive.ndim != 1:
            raise ValueError(
                "macro_proba_positive must have shape [batch_size]. "
                f"Got {macro_proba_positive.shape}."
            )

        self.macro_proba_positive = macro_proba_positive
        self.positive_class_name = str(positive_class_name)

    def probability(self, class_name: str | int) -> torch.Tensor:
        """
        Return the predicted probability for a given class.

        Parameters
        ----------
        class_name:
            Requested class label.

        Returns
        -------
        torch.Tensor
            Probability of the requested class for every sample.
        """
        class_name = str(class_name)

        if class_name == self.positive_class_name:
            return self.macro_proba_positive

        return 1.0 - self.macro_proba_positive