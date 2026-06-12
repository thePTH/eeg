import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from features.dataset import FeaturesDatasetSelector
from features.io import FeaturesDatasetIO

from prediction.decision_tree.base import (
    DecisionTree,
    DecisionTreeParameters,
)

from prediction.neural_network.dataset import (
    NeuroSymbolicEEGDataLoaderParameters,
    NeuroSymbolicEEGDataloaderFactory,
)

from prediction.neural_network.neural_backbone.model import (
    MultiScaleDeepEEGNet,
)

from prediction.neural_network.neuro_symbolic.rule_following_evaluator import (
    RuleEvaluator,
)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--models-dir", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="rule_evaluation_results.csv")

    parser.add_argument("--rule-activation-threshold", type=float, default=0.8)
    parser.add_argument("--prediction-threshold", type=float, default=0.5)

    parser.add_argument("--n-rules-to-keep", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)

    return parser.parse_args()


def parse_model_name(model_path: Path):
    pattern = r"^model_seed-(\d+)_lambda-([0-9]+(?:\.[0-9]+)?)\.pt$"

    match = re.match(
        pattern,
        model_path.name,
    )

    if match is None:
        raise ValueError(
            f"Invalid model filename: {model_path.name}"
        )

    seed = int(match.group(1))
    lambda_logic = float(match.group(2))

    return seed, lambda_logic


def load_dataset():
    dataset = FeaturesDatasetIO.load(
        "computed_features/dethamp",
        "raw_data",
    )

    dataset = dataset.selector.filter_by_healthstate(
        ["AD", "CN"]
    )

    dataset = FeaturesDatasetSelector.select(
        dataset,
        feature_family_names=[
            "theta_alpha_ratio",
            "spectral_power_ratio",
            "alpha",
            "beta",
            "gamma",
        ],
    )

    return dataset


def build_decision_tree():
    return DecisionTree(
        parameters=DecisionTreeParameters(
            criterion="gini",
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=10,
        )
    )


def load_model(model_path: Path):
    model = MultiScaleDeepEEGNet()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    args = parse_args()

    models_dir = Path(args.models_dir)
    output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_dataset()

    model_paths = sorted(
        models_dir.glob("model_seed-*_lambda-*.pt")
    )

    if len(model_paths) == 0:
        raise FileNotFoundError(
            f"No model found in {models_dir} with pattern "
            "`model_seed-*_lambda-*.pt`."
        )

    print(f"Found {len(model_paths)} models.")

    rows = []

    for model_path in model_paths:
        print("\n" + "=" * 80)
        print(f"Evaluating {model_path.name}")

        seed, lambda_logic = parse_model_name(
            model_path
        )

        model = load_model(
            model_path
        )

        decision_tree = build_decision_tree()

        dataloader_params = NeuroSymbolicEEGDataLoaderParameters(
            split_strategy="random",
            preprocessing_mode="mtdnet",
            random_seed=seed,
            decision_tree=decision_tree,
            batch_size=args.batch_size,
        )

        (
            rules,
            _,
            _,
            test_loader,
        ) = NeuroSymbolicEEGDataloaderFactory.build_all(
            features_dataset=dataset,
            params=dataloader_params,
        )

        rules = rules[: args.n_rules_to_keep]

        print(f"Seed: {seed}")
        print(f"Lambda logic: {lambda_logic}")
        print(f"Number of evaluated rules: {len(rules)}")
        print(f"Prediction threshold: {args.prediction_threshold}")
        print(f"Rule activation threshold: {args.rule_activation_threshold}")

        for i, rule in enumerate(rules):
            print(
                f"Rule {i} | "
                f"class={rule.predicted_class} | "
                f"score={rule.score:.4f} | "
                f"support={rule.support} | "
                f"p={rule.prediction_probability:.4f}"
            )

        evaluator = RuleEvaluator(
            threshold=args.prediction_threshold,
            rule_activation_threshold=args.rule_activation_threshold,
            macro_aggregation_method="mean_probability",
        )

        results = evaluator.evaluation(
            model=model,
            rules=rules,
            dataloader=test_loader,
        )

        row = {
            "model_path": str(model_path),
            "seed": seed,
            "lambda_logic": lambda_logic,
            "prediction_threshold": args.prediction_threshold,
            "rule_activation_threshold": args.rule_activation_threshold,
            **results.to_dict(),
        }

        rows.append(row)

        print(row)

    df = pd.DataFrame(rows)

    df = df.sort_values(
        by=["lambda_logic", "seed"],
    ).reset_index(drop=True)

    df.to_csv(
        output_csv,
        index=False,
    )

    print("\n" + "=" * 80)
    print(f"Results saved to {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()