from __future__ import annotations

import numpy as np
import torch

from eeg.data import EEGProcessedData
from features.dataset import SelectedFeaturesDataset


class TorchHelper:
    """
    Helper for converting project objects to PyTorch tensors compatible with
    the neuro-symbolic model.

    Conventions
    -----------
    x_raw
        Tensor with shape ``[batch_size, n_channels, n_times]``.

    x_feat
        Tensor with shape ``[n_samples, n_features]``.
    """

    @staticmethod
    def eeg_to_tensor(
        eegs: list[EEGProcessedData],
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Convert a list of EEGProcessedData objects into an ``x_raw`` batch.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[batch_size, n_channels, n_times]``.
        """
        if not eegs:
            raise ValueError("`eegs` cannot be empty.")

        arrays = []
        reference_shape = None

        for eeg in eegs:
            data = np.asarray(eeg.data, dtype=np.float32)

            if data.ndim != 2:
                raise ValueError(
                    f"Expected EEG data with shape [n_channels, n_times], "
                    f"got shape {data.shape}."
                )

            if reference_shape is None:
                reference_shape = data.shape
            elif data.shape != reference_shape:
                raise ValueError(
                    "All EEGProcessedData must have the same shape to build a batch. "
                    f"Expected {reference_shape}, got {data.shape}."
                )

            arrays.append(data)

        batch = np.stack(arrays, axis=0)

        return torch.as_tensor(
            batch,
            dtype=dtype,
            device=device,
        )

    @staticmethod
    def features_dataset_to_tensor(
        dataset: SelectedFeaturesDataset,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """
        Convert a SelectedFeaturesDataset into a PyTorch ``x_feat`` tensor.

        Returns
        -------
        torch.Tensor
            Tensor with shape ``[n_samples, n_features]``.
        """
        X = dataset.X

        non_numeric_columns = [
            col for col in X.columns
            if not np.issubdtype(X[col].dtype, np.number)
        ]

        if non_numeric_columns:
            raise TypeError(
                "SelectedFeaturesDataset.X contains non-numeric columns. "
                f"Non-numeric columns: {non_numeric_columns}"
            )

        values = X.to_numpy(dtype=np.float32, copy=True)

        if values.ndim != 2:
            raise ValueError(
                f"Expected shape [n_samples, n_features], got {values.shape}"
            )

        return torch.as_tensor(
            values,
            dtype=dtype,
            device=device,
        )


class MacroToMicroSegmenter:
    """
    Split a batch of macro EEG signals into micro-segments.

    Input
    -----
    macro_x_raw
        Tensor with shape ``[batch_size, n_channels, n_times_macro]``.

    Output
    ------
    micro_x_raw
        Tensor with shape
        ``[n_micro_segments, batch_size, n_channels, n_times_micro]``.
    """

    @staticmethod
    def split(macro_x_raw: torch.Tensor, n_micro_segments: int = 60) -> torch.Tensor:
        """
        Split macro EEG recordings into equally sized micro-segments.

        Parameters
        ----------
        macro_x_raw
            Input macro EEG tensor.
        n_micro_segments
            Number of micro-segments to create along the time axis.

        Returns
        -------
        torch.Tensor
            Segmented EEG tensor with shape
            ``[n_micro_segments, batch_size, n_channels, n_times_micro]``.
        """
        if macro_x_raw.ndim != 3:
            raise ValueError(
                "macro_x_raw must have shape [batch_size, n_channels, n_times_macro]. "
                f"Got {macro_x_raw.shape}."
            )

        batch_size, n_channels, n_times_macro = macro_x_raw.shape

        if n_times_macro % n_micro_segments != 0:
            raise ValueError(
                f"n_times_macro={n_times_macro} is not divisible by "
                f"n_micro_segments={n_micro_segments}."
            )

        n_times_micro = n_times_macro // n_micro_segments

        micro_x_raw = macro_x_raw.reshape(
            batch_size,
            n_channels,
            n_micro_segments,
            n_times_micro,
        )

        micro_x_raw = micro_x_raw.permute(2, 0, 1, 3).contiguous()

        return micro_x_raw