import argparse
import csv
from pathlib import Path

import numpy as np

from training.config import ExperimentConfig
from training.experiment import EEGExperimentRunner


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--split-strategy",
        type=str,
        choices=["mtdnet", "random"],
        default="random",
        help="Dataset splitting strategy.",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46],
        help="Seeds used for repeated grouped random validation.",
    )

    parser.add_argument(
        "--lambda-logic-values",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.3, 0.5],
        help="List of lambda_logic values to compare.",
    )

    return parser.parse_args()


def summarize(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values))


def main():
    args = parse_args()

    all_rows = []

    for lambda_logic in args.lambda_logic_values:

        lambda_metrics = []

        print("\n" + "=" * 80)
        print(f"Running lambda_logic = {lambda_logic}")
        print("=" * 80)

        for seed in args.seeds:

            config = ExperimentConfig(
                lambda_logic=lambda_logic,
                epochs=args.epochs,
                random_seed=seed,
                split_strategy=args.split_strategy,
            )

            runner = EEGExperimentRunner(config)

            _, _, test_metrics = runner.run()

            row = {
                "lambda_logic": lambda_logic,
                "seed": seed,
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

        print("\nSummary for lambda_logic =", lambda_logic)
        print(f"Balanced accuracy: {bal_mean:.4f} ± {bal_std:.4f}")
        print(f"F1-score:          {f1_mean:.4f} ± {f1_std:.4f}")

    output_dir = Path("runs") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "lambda_logic_comparison.csv"

    # Détection automatique de toutes les colonnes présentes
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
    print("Final comparison")
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

    print(f"\nDetailed results saved to: {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()