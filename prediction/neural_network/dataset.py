from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from features.dataset import SelectedFeaturesDataset
from prediction.decision_tree.tunning import DecisionTree
from prediction.neural_network.helpers import MacroToMicroSegmenter
from rules.differentiable_rule import (
    DifferentiableDecisionRule,
    DifferentiableDecisionRulesFactory,
)

SplitMode = Literal["train", "val", "test"]
PreprocessingMode = Literal["mtdnet", "raw"]
SplitStrategy = Literal["mtdnet", "random"]


# =============================================================================
# MTDNet subject-independent split
# =============================================================================

class MTDNetSubjectSplitEngine:
    VALID_SPLIT_MODES = {"train", "val", "test"}

    @staticmethod
    def normalize_subject_id(subject_id: str | int) -> str:
        value = str(subject_id).strip()

        if value.startswith("sub-"):
            value = value.replace("sub-", "")

        if value.isdigit():
            return f"sub-{int(value):03d}"

        return str(subject_id).strip()

    @staticmethod
    def get_split_subjects(
        dataset_name: str = "miltiadous",
        task: str = "hc-ad",
    ) -> dict[str, list[str]]:

        dataset_name = dataset_name.lower().strip()
        task = task.lower().strip()

        if "miltiadous" not in dataset_name or task != "hc-ad":
            raise ValueError(
                "MTDNet split currently implemented only for "
                "dataset_name='miltiadous' and task='hc-ad'."
            )

        train_ids = [
            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            50, 51, 52, 53,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            16, 17, 18, 19, 20, 21,
        ]

        val_ids = [
            54, 55, 56, 57, 58, 59,
            22, 23, 24, 25, 26, 27, 28,
        ]

        test_ids = [
            60, 61, 62, 63, 64, 65,
            29, 30, 31, 32, 33, 34, 35, 36,
        ]

        return {
            "train": [f"sub-{i:03d}" for i in train_ids],
            "val": [f"sub-{i:03d}" for i in val_ids],
            "test": [f"sub-{i:03d}" for i in test_ids],
        }

    @classmethod
    def split(
        cls,
        features_dataset: SelectedFeaturesDataset,
        split_mode: SplitMode,
        dataset_name: str = "miltiadous",
        task: str = "hc-ad",
    ) -> SelectedFeaturesDataset:

        if split_mode not in cls.VALID_SPLIT_MODES:
            raise ValueError(
                "`split_mode` must be one of "
                f"{sorted(cls.VALID_SPLIT_MODES)}. "
                f"Got {split_mode}."
            )

        split_subjects = set(
            cls.get_split_subjects(
                dataset_name=dataset_name,
                task=task,
            )[split_mode]
        )

        row_indices = [
            index
            for index, subject_id in enumerate(
                features_dataset.subject_dataframe["subject_id"]
            )
            if cls.normalize_subject_id(subject_id) in split_subjects
        ]

        if not row_indices:
            available_subjects = [
                cls.normalize_subject_id(subject_id)
                for subject_id in features_dataset.subject_dataframe["subject_id"]
            ]

            raise ValueError(
                f"MTDNet split '{split_mode}' produced an empty dataset. "
                f"Available subjects start with: {available_subjects[:10]}"
            )

        return features_dataset.select_rows(row_indices)

    @classmethod
    def split_all(
        cls,
        features_dataset: SelectedFeaturesDataset,
        dataset_name: str = "miltiadous",
        task: str = "hc-ad",
    ) -> tuple[
        SelectedFeaturesDataset,
        SelectedFeaturesDataset,
        SelectedFeaturesDataset,
    ]:

        train_dataset = cls.split(
            features_dataset=features_dataset,
            split_mode="train",
            dataset_name=dataset_name,
            task=task,
        )

        val_dataset = cls.split(
            features_dataset=features_dataset,
            split_mode="val",
            dataset_name=dataset_name,
            task=task,
        )

        test_dataset = cls.split(
            features_dataset=features_dataset,
            split_mode="test",
            dataset_name=dataset_name,
            task=task,
        )

        return train_dataset, val_dataset, test_dataset


# =============================================================================
# Parameters
# =============================================================================

@dataclass(frozen=True)
class EEGAugmentationParameters:
    p_time_flip: float = 0.5
    p_channel_shuffle: float = 0.5
    p_time_mask: float = 0.5
    time_mask_ratio: float = 0.2


@dataclass(frozen=True)
class NeuroSymbolicEEGDataLoaderParameters:
    batch_size: int = 8
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False

    preprocessing_mode: PreprocessingMode = "mtdnet"
    n_micro_segments: int = 60

    augmentation_params: EEGAugmentationParameters = field(
        default_factory=EEGAugmentationParameters
    )

    # ==========================================================
    # Split strategy
    # ==========================================================

    split_strategy: SplitStrategy = "mtdnet"

    random_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.3

    mtdnet_dataset_name: str = "miltiadous"
    mtdnet_task: str = "hc-ad"

    # ==========================================================
    # Rule extraction
    # ==========================================================

    decision_tree: DecisionTree | None = None
    c_tau: float = 0.1
    min_tau: float = 0.001


# =============================================================================
# Dataset
# =============================================================================

class NeuroSymbolicEEGDataset(Dataset):

    TARGET_MAPPING = {
        "Healthy": 0.0,
        "Alzheimer": 1.0,
    }

    VALID_SPLIT_MODES = {"train", "val", "test"}
    VALID_PREPROCESSING_MODES = {"mtdnet", "raw"}

    def __init__(
        self,
        features_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters,
        split_mode: SplitMode,
    ):
        if features_dataset is None:
            raise ValueError("`features_dataset` cannot be None.")

        if params is None:
            raise ValueError("`params` cannot be None.")

        if split_mode not in self.VALID_SPLIT_MODES:
            raise ValueError(
                "`split_mode` must be one of "
                f"{sorted(self.VALID_SPLIT_MODES)}. "
                f"Got {split_mode}."
            )

        if params.preprocessing_mode not in self.VALID_PREPROCESSING_MODES:
            raise ValueError(
                "`preprocessing_mode` must be one of "
                f"{sorted(self.VALID_PREPROCESSING_MODES)}. "
                f"Got {params.preprocessing_mode}."
            )

        if params.n_micro_segments <= 0:
            raise ValueError(
                "`n_micro_segments` must be strictly positive. "
                f"Got {params.n_micro_segments}."
            )

        if len(features_dataset.eegs) != len(features_dataset.X):
            raise ValueError(
                "`features_dataset.eegs` and "
                "`features_dataset.X` must have same length."
            )

        self.features_dataset = features_dataset
        self.eegs = features_dataset.eegs
        self.params = params
        self.split_mode = split_mode

        self._x_feat = self._build_x_feat_tensor(features_dataset)
        self._y = self._build_y_tensor(features_dataset)

    def __len__(self) -> int:
        return len(self.eegs)

    def __getitem__(self, index: int):
        eeg = self.eegs[index]

        macro_x_raw_np = np.asarray(
            eeg.data,
            dtype=np.float32,
        )

        if macro_x_raw_np.ndim != 2:
            raise ValueError(
                "Each EEG must have shape [n_channels, n_times]. "
                f"Got shape {macro_x_raw_np.shape} at index {index}."
            )

        macro_x_raw = torch.from_numpy(macro_x_raw_np)

        micro_x_raws = MacroToMicroSegmenter.split(
            macro_x_raw.unsqueeze(0),
            n_micro_segments=self.params.n_micro_segments,
        ).squeeze(1)

        if self.params.preprocessing_mode == "mtdnet":
            micro_x_raws = self._preprocess_micro_segments(micro_x_raws)

            if self.split_mode == "train":
                micro_x_raws = self._augment_micro_segments(micro_x_raws)

        elif self.params.preprocessing_mode == "raw":
            pass

        else:
            raise RuntimeError(
                "Unexpected preprocessing_mode: "
                f"{self.params.preprocessing_mode}"
            )

        micro_x_raws = micro_x_raws.contiguous().float()
        macro_x_feat = self._x_feat[index]
        y_true = self._y[index]

        return micro_x_raws, macro_x_feat, y_true

    @staticmethod
    def _preprocess_micro_segments(
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:

        channel_mean = micro_x_raws.mean(dim=2, keepdim=True)
        x = micro_x_raws - channel_mean

        global_mean = x.mean(dim=(1, 2), keepdim=True)
        global_std = x.std(dim=(1, 2), keepdim=True)

        x = torch.where(
            global_std < 1e-8,
            torch.zeros_like(x),
            (x - global_mean) / global_std.clamp_min(1e-8),
        )

        return x.float()

    def _augment_micro_segments(
        self,
        micro_x_raws: torch.Tensor,
    ) -> torch.Tensor:

        params = self.params.augmentation_params
        x = micro_x_raws.clone()

        n_micro_segments, n_channels, n_times = x.shape

        for i in range(n_micro_segments):

            if torch.rand(()) < params.p_time_flip:
                x[i] = torch.flip(x[i], dims=[1])

            if torch.rand(()) < params.p_channel_shuffle:
                permutation = torch.randperm(n_channels)
                x[i] = x[i, permutation, :]

            if torch.rand(()) < params.p_time_mask:
                n_masked_times = int(n_times * params.time_mask_ratio)

                if n_masked_times > 0:
                    masked_indices = torch.randperm(n_times)[:n_masked_times]
                    x[i, :, masked_indices] = 0.0

        return x

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
                "SelectedFeaturesDataset.X contains "
                f"non-numeric columns: {non_numeric_columns}"
            )

        values = X.to_numpy(
            dtype=np.float32,
            copy=True,
        )

        return torch.as_tensor(
            values,
            dtype=torch.float32,
        )

    @classmethod
    def _build_y_tensor(
        cls,
        features_dataset: SelectedFeaturesDataset,
    ) -> torch.Tensor:

        y = features_dataset.y.to_numpy()

        unknown_labels = [
            label
            for label in set(y)
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

        return torch.as_tensor(
            y_values,
            dtype=torch.float32,
        )


# =============================================================================
# Dataloader factory
# =============================================================================

class NeuroSymbolicEEGDataloaderFactory:

    @staticmethod
    def _split_dataset(
        features_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters,
    ) -> tuple[
        SelectedFeaturesDataset,
        SelectedFeaturesDataset,
        SelectedFeaturesDataset,
    ]:

        if params.split_strategy == "mtdnet":
            return MTDNetSubjectSplitEngine.split_all(
                features_dataset=features_dataset,
                dataset_name=params.mtdnet_dataset_name,
                task=params.mtdnet_task,
            )

        if params.split_strategy == "random":
            return features_dataset.selector.group_train_val_test_split(
                random_state=params.random_seed,
                test_size=params.test_size,
                val_size=params.val_size,
            )

        raise ValueError(
            "`split_strategy` must be either 'mtdnet' or 'random'. "
            f"Got {params.split_strategy}."
        )

    @staticmethod
    def _extract_rules_from_train_dataset(
        train_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters,
    ) -> list[DifferentiableDecisionRule]:

        if params.decision_tree is None:
            raise ValueError(
                "`params.decision_tree` cannot be None when calling "
                "`build_all`, because rules must be extracted from a tree "
                "trained on the train split."
            )
        
        trained_tree = params.decision_tree.train(train_dataset)

        differentiable_rules, _ = DifferentiableDecisionRulesFactory.build(
            trained_tree,
            c_tau=params.c_tau,
            min_tau=params.min_tau,
        )

        rules = sorted(
            differentiable_rules,
            key=lambda rule: rule.score,
            reverse=True,
        )

        return rules

    @staticmethod
    def build(
        features_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters | None = None,
        split_mode: SplitMode = "train",
    ) -> DataLoader:
        """
        Construit uniquement un DataLoader.

        Important
        ---------
        Cette méthode ne fait aucun split.
        Le dataset reçu doit déjà correspondre au split souhaité.
        """

        if params is None:
            params = NeuroSymbolicEEGDataLoaderParameters()

        torch_dataset = NeuroSymbolicEEGDataset(
            features_dataset=features_dataset,
            params=params,
            split_mode=split_mode,
        )

        return DataLoader(
            torch_dataset,
            batch_size=params.batch_size,
            shuffle=params.shuffle if split_mode == "train" else False,
            num_workers=params.num_workers,
            drop_last=params.drop_last,
            pin_memory=params.pin_memory,
        )

    @staticmethod
    def build_all(
        features_dataset: SelectedFeaturesDataset,
        params: NeuroSymbolicEEGDataLoaderParameters | None = None,
    ) -> tuple[
        list[DifferentiableDecisionRule],
        DataLoader,
        DataLoader,
        DataLoader,
    ]:
        """
        Pipeline complet :

        1. Split du SelectedFeaturesDataset.
        2. Entraînement de l'arbre sur train_dataset uniquement.
        3. Extraction des règles différentiables.
        4. Construction des DataLoaders.

        Retourne
        --------
        rules, train_loader, val_loader, test_loader
        """

        if params is None:
            params = NeuroSymbolicEEGDataLoaderParameters()

        train_dataset, val_dataset, test_dataset = (
            NeuroSymbolicEEGDataloaderFactory._split_dataset(
                features_dataset=features_dataset,
                params=params,
            )
        )

        trained_tree = params.decision_tree.train(train_dataset)

        differentiable_rules, _ = DifferentiableDecisionRulesFactory.build(
            trained_tree,
            c_tau=params.c_tau,
            min_tau=params.min_tau,
        )

        rules = sorted(
            differentiable_rules,
            key=lambda rule: rule.score,
            reverse=True,
        )

        train_loader = NeuroSymbolicEEGDataloaderFactory.build(
            features_dataset=train_dataset,
            params=params,
            split_mode="train",
        )

        val_loader = NeuroSymbolicEEGDataloaderFactory.build(
            features_dataset=val_dataset,
            params=params,
            split_mode="val",
        )

        test_loader = NeuroSymbolicEEGDataloaderFactory.build(
            features_dataset=test_dataset,
            params=params,
            split_mode="test",
        )

        return rules, train_loader, val_loader, test_loader