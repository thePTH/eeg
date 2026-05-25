import argparse

from training.config import ExperimentConfig
from training.experiment import EEGExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lambda-logic",
        type=float,
        default=0.0,
        help="Weight of the logic loss in the total loss.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = ExperimentConfig(
        lambda_logic=args.lambda_logic,
        epochs=args.epochs,
        random_seed=args.seed,
    )

    runner = EEGExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()