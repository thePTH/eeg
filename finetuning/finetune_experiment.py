from pathlib import Path

import torch

from training.config import ExperimentConfig
from training.experiment import EEGExperimentRunner


class EEGFineTuningRunner(EEGExperimentRunner):

    def __init__(
        self,
        config: ExperimentConfig,
        pretrained_model_path: str | Path,
    ):
        super().__init__(config)

        self.pretrained_model_path = Path(pretrained_model_path)

        self.run_name = (
            f"{config.experiment_name}"
            f"_finetune"
            f"_split_{config.split_strategy}"
            f"_lambda_{config.lambda_logic}"
            f"_lr_{config.lr}"
            f"_wd_{config.weight_decay}"
            f"_seed_{config.random_seed}"
        )

        self.run_dir = Path("runs") / "finetuning" / self.run_name

        self.log_dir = self.run_dir / "logs"
        self.output_dir = self.run_dir / "outputs"
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.logger = self._build_logger()

    def build_trainer(self):

        trainer = super().build_trainer()

        trainer.params.tensorboard_log_dir = str(
            Path("runs")
            / "finetuning"
            / "tensorboard"
            / self.run_name
        )

        return trainer

    def load_pretrained_weights(
        self,
        model,
        device,
    ):

        if not self.pretrained_model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found: {self.pretrained_model_path}"
            )

        state_dict = torch.load(
            self.pretrained_model_path,
            map_location=device,
        )

        model.load_state_dict(state_dict)

        self.logger.info(
            "Loaded pretrained model from %s",
            self.pretrained_model_path,
        )

        return model

    def run(self):

        self.logger.info("=" * 80)
        self.logger.info("Starting fine-tuning run")
        self.logger.info("Run name: %s", self.run_name)
        self.logger.info("Pretrained model: %s", self.pretrained_model_path)
        self.logger.info("Split strategy: %s", self.config.split_strategy)
        self.logger.info("Lambda logic: %.4f", self.config.lambda_logic)
        self.logger.info("Learning rate: %.2e", self.config.lr)
        self.logger.info("Weight decay: %.2e", self.config.weight_decay)
        self.logger.info("Epochs: %d", self.config.epochs)
        self.logger.info("Seed: %d", self.config.random_seed)
        self.logger.info("=" * 80)

        self.set_seed()

        device = self.get_device()
        dataset = self.load_dataset()

        (
            rules,
            train_loader,
            val_loader,
            test_loader,
        ) = self.build_dataloaders_and_rules(dataset)

        model = self.build_model(device)

        model = self.load_pretrained_weights(
            model=model,
            device=device,
        )

        trainer = self.build_trainer()

        model, history = trainer.train(
            model=model,
            rules=rules,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            return_history=True,
        )

        test_metrics = trainer.evaluate(
            model=model,
            rules=rules,
            dataloader=test_loader,
        )

        self.save_outputs(
            model=model,
            history=history,
        )

        self.logger.info("Test metrics: %s", test_metrics)
        self.logger.info("Fine-tuning finished successfully")

        return model, history, test_metrics