from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from eeg.data import EEGProcessedData
from features.dataset import SelectedFeaturesDataset


@dataclass(frozen=True)
class NeuroSymbolicEEGDataLoaderParameters:
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False





@dataclass(frozen=True)
class NeuroSymbolicEEGDataLoaderParameters:
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False


class NeuroSymbolicEEGDataset(Dataset):
    """
    Dataset PyTorch retournant :
        macro_x_raw  : [n_channels, n_times]
        macro_x_feat : [n_features]
        y_true       : scalaire float

    Les EEG sont chargés lazy dans __getitem__.
    """

    TARGET_MAPPING = {
        "Healthy": 0.0,
        "Alzheimer": 1.0,
    }

    def __init__(
        self,
        features_dataset: SelectedFeaturesDataset,
    ):
        if features_dataset is None:
            raise ValueError("`features_dataset` cannot be None.")

        if len(features_dataset.eegs) != len(features_dataset.X):
            raise ValueError(
                "`features_dataset.eegs` and `features_dataset.X` must have "
                "the same length."
            )

        self.features_dataset = features_dataset
        self.eegs = features_dataset.eegs

        self._x_feat = self._build_x_feat_tensor(features_dataset)
        self._y = self._build_y_tensor(features_dataset)

    def __len__(self) -> int:
        return len(self.eegs)

    def __getitem__(self, index: int):
        eeg = self.eegs[index]

        # Charge uniquement cet EEG, puis le libère si nécessaire grâce à EEGData.data
        raw_data = np.asarray(eeg.data, dtype=np.float32)

        if raw_data.ndim != 2:
            raise ValueError(
                "Each EEG must have shape [n_channels, n_times]. "
                f"Got shape {raw_data.shape} at index {index}."
            )

        macro_x_raw = torch.as_tensor(raw_data, dtype=torch.float32)
        macro_x_feat = self._x_feat[index]
        y_true = self._y[index]

        return macro_x_raw, macro_x_feat, y_true

    @staticmethod
    def _build_x_feat_tensor(
        features_dataset: SelectedFeaturesDataset,
    ) -> torch.Tensor:
        X = features_dataset.X

        non_numeric_columns = [
            col
            for col in X.columns
            if not np.issubdtype(X[col].dtype, np.number)
        ]

        if non_numeric_columns:
            raise TypeError(
                "SelectedFeaturesDataset.X contains non-numeric columns: "
                f"{non_numeric_columns}"
            )

        values = X.to_numpy(dtype=np.float32, copy=True)
        return torch.as_tensor(values, dtype=torch.float32)

    @classmethod
    def _build_y_tensor(
        cls,
        features_dataset: SelectedFeaturesDataset,
    ) -> torch.Tensor:
        y = features_dataset.y.to_numpy()

        unknown_labels = [
            label for label in set(y)
            if label not in cls.TARGET_MAPPING
        ]

        if unknown_labels:
            raise ValueError(
                "Unexpected labels in subject_health. "
                f"Expected only {list(cls.TARGET_MAPPING.keys())}, "
                f"got {unknown_labels}."
            )

        y_values = np.array(
            [cls.TARGET_MAPPING[label] for label in y],
            dtype=np.float32,
        )

        return torch.as_tensor(y_values, dtype=torch.float32)


class NeuroSymbolicEEGDataloaderFactory:

    @staticmethod
    def build(
        features_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters | None = None,
    ) -> DataLoader:
        if params is None:
            params = NeuroSymbolicEEGDataLoaderParameters()

        torch_dataset = NeuroSymbolicEEGDataset(
            features_dataset=features_dataset,
        )

        return DataLoader(
            torch_dataset,
            batch_size=params.batch_size,
            shuffle=params.shuffle,
            num_workers=params.num_workers,
            drop_last=params.drop_last,
            pin_memory=params.pin_memory,
        )