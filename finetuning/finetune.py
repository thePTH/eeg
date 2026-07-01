import argparse
import csv
from pathlib import Path

import numpy as np

from training.config import ExperimentConfig
from finetuning.finetune_experiment import EEGFineTuningRunner


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of additional fine-tuning epochs.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate used during fine-tuning.",
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay used during fine-tuning.",
    )

    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["mtdnet", "random"],
        default="random",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
    )

    parser.add_argument(
        "--lambda-logic-values",
        type=float,
        nargs="+",
        default=[0.6, 0.7, 0.8, 1.0],
    )

    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        default="pretrained_models",
    )

    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float]:

    return (
        float(np.mean(values)),
        float(np.std(values)),
    )


def find_pretrained_model(
    pretrained_dir: Path,
    seed: int,
) -> Path:

    return pretrained_dir / f"model_{seed}.pt"


def main():

    args = parse_args()

    pretrained_dir = Path(args.pretrained_model_dir)

    all_rows = []

    for lambda_logic in args.lambda_logic_values:

        lambda_metrics = []

        print("\n" + "=" * 80)
        print(f"Fine-tuning lambda={lambda_logic}")
        print(f"lr={args.lr} | weight_decay={args.weight_decay}")
        print("=" * 80)

        for seed in args.seeds:

            pretrained_model_path = find_pretrained_model(
                pretrained_dir=pretrained_dir,
                seed=seed,
            )

            if not pretrained_model_path.exists():
                raise FileNotFoundError(
                    f"Pretrained model not found: "
                    f"{pretrained_model_path}"
                )

            config = ExperimentConfig(
                epochs=args.epochs,
                random_seed=seed,
                split_strategy=args.split_strategy,
                lambda_logic=lambda_logic,
                lr=args.lr,
                weight_decay=args.weight_decay,
            )

            runner = EEGFineTuningRunner(
                config=config,
                pretrained_model_path=pretrained_model_path,
            )

            _, _, test_metrics = runner.run()

            row = {
                "lambda_logic": lambda_logic,
                "seed": seed,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "pretrained_model_path": str(pretrained_model_path),
                **test_metrics,
            }

            all_rows.append(row)
            lambda_metrics.append(test_metrics)

        balanced_accs = [
            metric["test_balanced_accuracy"]
            for metric in lambda_metrics
        ]

        f1_scores = [
            metric["test_f1_score"]
            for metric in lambda_metrics
        ]

        bal_mean, bal_std = summarize(balanced_accs)
        f1_mean, f1_std = summarize(f1_scores)

        print(f"\nLambda {lambda_logic}")
        print(f"Balanced accuracy : {bal_mean:.4f} ± {bal_std:.4f}")
        print(f"F1-score          : {f1_mean:.4f} ± {f1_std:.4f}")

    output_dir = Path("runs") / "finetuning_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "finetuning_results.csv"

    fieldnames = sorted(
        {
            key
            for row in all_rows
            for key in row.keys()
        }
    )

    with open(
        csv_path,
        mode="w",
        newline="",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print("\n" + "=" * 80)
    print("Fine-tuning comparison finished")
    print("=" * 80)

    for lambda_logic in args.lambda_logic_values:

        rows = [
            row
            for row in all_rows
            if row["lambda_logic"] == lambda_logic
        ]

        balanced_accs = [
            row["test_balanced_accuracy"]
            for row in rows
        ]

        f1_scores = [
            row["test_f1_score"]
            for row in rows
        ]

        bal_mean, bal_std = summarize(balanced_accs)
        f1_mean, f1_std = summarize(f1_scores)

        print(
            f"lambda={lambda_logic:<5} | "
            f"balanced_acc={bal_mean:.4f} ± {bal_std:.4f} | "
            f"f1={f1_mean:.4f} ± {f1_std:.4f}"
        )

    print(f"\nResults saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()