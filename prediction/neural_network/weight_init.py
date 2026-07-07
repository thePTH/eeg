from __future__ import annotations

import torch.nn as nn


class EEGWeightInitializer:
    """Utility class for initializing neural-network weights."""

    @staticmethod
    def apply(
        model: nn.Module,
        method: str = "kaiming",
    ) -> nn.Module:
        """
        Apply the requested initialization method to all modules in a model.

        Parameters
        ----------
        model:
            PyTorch model to initialize.
        method:
            Weight initialization strategy. Supported values are:
            - "kaiming"
            - "xavier"
            - "orthogonal"

        Returns
        -------
        nn.Module
            The initialized model.
        """

        def init_fn(module):
            EEGWeightInitializer.initialize_module(
                module=module,
                method=method,
            )

        model.apply(init_fn)

        return model

    @staticmethod
    def initialize_module(
        module: nn.Module,
        method: str = "kaiming",
    ) -> None:
        """
        Initialize a single PyTorch module.

        Supported modules
        -----------------
        - Conv1d
        - Linear
        - BatchNorm1d
        - LSTM
        """

        # ==========================================================
        # Conv1D / Linear
        # ==========================================================
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            if method == "kaiming":
                nn.init.kaiming_normal_(
                    module.weight,
                    nonlinearity="relu",
                )

            elif method == "xavier":
                nn.init.xavier_normal_(module.weight)

            elif method == "orthogonal":
                nn.init.orthogonal_(module.weight)

            else:
                raise ValueError(
                    f"Unknown initialization method: {method}"
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # ==========================================================
        # Batch Normalization
        # ==========================================================
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        # ==========================================================
        # LSTM
        # ==========================================================
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():

                if "weight_ih" in name:
                    if method == "kaiming":
                        nn.init.kaiming_normal_(param)

                    elif method == "xavier":
                        nn.init.xavier_uniform_(param)

                    elif method == "orthogonal":
                        nn.init.orthogonal_(param)

                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)

                elif "bias" in name:
                    nn.init.zeros_(param)

                    # Initialize the forget gate bias to one,
                    # following the common LSTM initialization practice.
                    hidden_size = module.hidden_size
                    param.data[
                        hidden_size : 2 * hidden_size
                    ].fill_(1.0)