from pathlib import Path
import logging
import random

import numpy as np
import torch

from training.config import ExperimentConfig

from features.dataset import FeaturesDatasetSelector
from features.io import FeaturesDatasetIO

from prediction.decision_tree.base import DecisionTree, DecisionTreeParameters
from prediction.neural_network.dataset import (
    NeuroSymbolicEEGDataloaderFactory,
    NeuroSymbolicEEGDataLoaderParameters,
)
from prediction.neural_network.neural_backbone.model import MultiScaleDeepEEGNet
from prediction.neural_network.neuro_symbolic.trainer import (
    NeuroSymbolicDeepEEGTrainer,
    NeuroSymbolicDeepEEGTrainerParameters,
)
from prediction.neural_network.weight_init import EEGWeightInitializer


class EEGExperimentRunner:
    def __init__(self, config: ExperimentConfig):
        self.config = config

        self.run_name = (
            f"{config.experiment_name}"
            f"_split_{config.split_strategy}"
            f"_lambda_{config.lambda_logic}"
            f"_seed_{config.random_seed}"
        )

        self.run_dir = Path("runs") / self.run_name
        self.log_dir = self.run_dir / "logs"
        self.output_dir = self.run_dir / "outputs"
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.run_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            self.log_dir / "training.log",
            mode="w",
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)

        return logger

    def set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_seed)

        self.logger.info("Random seed fixed to %d", self.config.random_seed)

    def get_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.logger.info("Using device: %s", device)

        if device.type == "cuda":
            self.logger.info("GPU name: %s", torch.cuda.get_device_name(0))

        return device

    def load_dataset(self):
        self.logger.info("Loading dataset")

        dataset = FeaturesDatasetIO.load(
            self.config.dataset_folder,
            self.config.dataset_name,
        )

        dataset = dataset.selector.filter_by_healthstate(
            list(self.config.target_health_states)
        )

        dataset = FeaturesDatasetSelector.select(
            dataset,
            feature_family_names=list(self.config.feature_family_names),
        )

        self.logger.info("Dataset loaded and selected")

        return dataset

    def build_decision_tree(self) -> DecisionTree:
        decision_tree = DecisionTree(
            parameters=DecisionTreeParameters(
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=10,
            )
        )

        self.logger.info("Decision tree initialized")

        return decision_tree

    def build_dataloaders_and_rules(self, dataset):
        decision_tree = self.build_decision_tree()

        params = NeuroSymbolicEEGDataLoaderParameters(
            batch_size=self.config.batch_size,
            preprocessing_mode=self.config.preprocessing_mode,
            split_strategy=self.config.split_strategy,
            random_seed=self.config.random_seed,
            test_size=self.config.test_size,
            val_size=self.config.val_size,
            mtdnet_dataset_name=self.config.mtdnet_dataset_name,
            mtdnet_task=self.config.mtdnet_task,
            decision_tree=decision_tree,
        )

        rules, train_loader, val_loader, test_loader = (
            NeuroSymbolicEEGDataloaderFactory.build_all(
                features_dataset=dataset,
                params=params,
            )
        )

        rules = rules[: self.config.n_rules_to_keep]

        self.logger.info("Dataloaders created")
        self.logger.info("Number of selected rules: %d", len(rules))

        return rules, train_loader, val_loader, test_loader

    def build_model(self, device: torch.device):
        model = MultiScaleDeepEEGNet()
        model = EEGWeightInitializer.apply(model, method="kaiming")
        model = model.to(device)

        self.logger.info("Model initialized")

        return model

    def build_trainer(self):
        params = NeuroSymbolicDeepEEGTrainerParameters(
            epochs=self.config.epochs,
            lambda_logic=self.config.lambda_logic,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            macro_aggregation_method=self.config.macro_aggregation_method,
            supervised_loss_compute_method=self.config.supervised_loss_compute_method,
            tensorboard_log_dir=str(self.run_dir / "tensorboard"),
        )

        return NeuroSymbolicDeepEEGTrainer(params)

    def save_outputs(self, model, history) -> None:
        model_path = self.checkpoint_dir / "model.pt"
        history_path = self.output_dir / "history.npy"

        torch.save(model.state_dict(), model_path)
        np.save(history_path, history, allow_pickle=True)

        self.logger.info("Model saved to %s", model_path)
        self.logger.info("History saved to %s", history_path)

    def run(self):
        self.logger.info("=" * 80)
        self.logger.info("Starting run: %s", self.run_name)
        self.logger.info("Split strategy: %s", self.config.split_strategy)
        self.logger.info("Lambda logic: %.4f", self.config.lambda_logic)
        self.logger.info("=" * 80)

        self.set_seed()
        device = self.get_device()

        dataset = self.load_dataset()

        rules, train_loader, val_loader, test_loader = (
            self.build_dataloaders_and_rules(dataset)
        )

        model = self.build_model(device)
        trainer = self.build_trainer()

        model, history = trainer.train(
            model=model,
            rules=rules,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            return_history=True,
        )

        self.save_outputs(model, history)

        self.logger.info("Run finished successfully")

        return model, history