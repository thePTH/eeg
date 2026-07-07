# EEG Neuro-Symbolic Alzheimer Prediction Project

This repository implements an end-to-end EEG analysis and neuro-symbolic learning pipeline for Alzheimer-vs-control classification. It covers raw EEG manipulation, preprocessing, feature extraction, statistical analysis, decision-tree rule extraction, differentiable rule conversion, neural-network training, fine-tuning, and post-hoc rule-following evaluation.

The project is designed around one central idea: use interpretable EEG biomarkers to extract symbolic decision rules, then use these rules as a differentiable regularization term during neural-network training. In practice, a model is trained both to predict the clinical class and to remain coherent with rules extracted from EEG features.

---

## Table of contents

1. [Repository overview](#repository-overview)
2. [Installation](#installation)
3. [Data organization](#data-organization)
4. [Core concepts](#core-concepts)
5. [EEG data manipulation](#eeg-data-manipulation)
6. [Preprocessing pipelines](#preprocessing-pipelines)
7. [Feature extraction](#feature-extraction)
8. [Feature datasets](#feature-datasets)
9. [Statistical analysis](#statistical-analysis)
10. [Decision trees and symbolic rules](#decision-trees-and-symbolic-rules)
11. [Differentiable rules](#differentiable-rules)
12. [Neural-network datasets and micro-segmentation](#neural-network-datasets-and-micro-segmentation)
13. [Neural backbones](#neural-backbones)
14. [Neuro-symbolic training](#neuro-symbolic-training)
15. [Command-line training](#command-line-training)
16. [Command-line fine-tuning](#command-line-fine-tuning)
17. [Command-line rule-following evaluation](#command-line-rule-following-evaluation)
18. [Outputs and experiment folders](#outputs-and-experiment-folders)
19. [TensorBoard](#tensorboard)
20. [Typical experiment workflows](#typical-experiment-workflows)
21. [Extending the project](#extending-the-project)
22. [Troubleshooting](#troubleshooting)

---

## Repository overview

```text
.
├── constants.py
├── eeg/                         # EEG objects, raw/processed data wrappers, signal analysis helpers
├── preprocessing/               # Preprocessing pipeline and individual MNE-compatible steps
├── features/                    # Feature definitions, extraction engines, feature dataset I/O
├── maths/                       # Low-level signal, spectral, wavelet, entropy, fractal and complexity tools
├── participants/                # Participant metadata, gender/group/health-state abstractions
├── stats/                       # Statistical queries, statistical tests, FDR correction, result containers
├── prediction/
│   ├── decision_tree/           # Decision-tree training, scoring, tuning, interpretation and visualization
│   └── neural_network/          # PyTorch datasets, neural backbones, trainer, evaluation utilities
├── rules/                       # Symbolic rules, differentiable rules, rule extraction from decision trees
├── training/                    # From-scratch neuro-symbolic training entry point
├── finetuning/                  # Fine-tuning entry point from pretrained models
├── evaluation/                  # Rule-following evaluation entry point
├── pretrained_models/           # Pretrained weights used by fine-tuning scripts
├── runs*/                       # Existing experiment outputs, logs, histories and checkpoints
├── papers/                      # Reference papers
└── requirements.txt
```

The most important terminal entry points are:

```bash
PYTHONPATH=. python training/train.py ...
PYTHONPATH=. python finetuning/finetune.py ...
PYTHONPATH=. python evaluation/evaluate_rule_following.py ...
```

Because the repository does not expose a package installation file such as `setup.py` or `pyproject.toml`, commands should be launched from the repository root with `PYTHONPATH=.` so Python can resolve imports such as `from training.config import ExperimentConfig`.

---

## Installation

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies include:

- `numpy`, `scipy`, `pandas` for numerical computing and tabular data;
- `mne`, `mne-bids`, `mne-connectivity`, `asrpy` for EEG manipulation and preprocessing;
- `PyWavelets` for wavelet features;
- `scikit-learn` for decision trees, metrics and train/validation/test splits;
- `torch` and `tensorboard` for neural-network training;
- `statsmodels` for multiple-comparison correction and statistical tests;
- `matplotlib`, `networkx` for plotting and graph utilities.

### 3. Check that the project imports correctly

Run from the repository root:

```bash
PYTHONPATH=. python - <<'PY'
from training.config import ExperimentConfig
from training.experiment import EEGExperimentRunner
print(ExperimentConfig())
PY
```

---

## Data organization

The training configuration expects an already computed feature dataset by default:

```python
ExperimentConfig(
    dataset_folder="computed_features/dethamp",
    dataset_name="raw_data",
)
```

This means that, at runtime, the project expects a folder equivalent to:

```text
computed_features/
└── dethamp/
    └── raw_data/
        ├── sub-001...
        ├── sub-002...
        └── ...
```

Each participant folder is loaded by `SingleParticipantProcessedFeatureDatasetIO.load()` and typically contains:

```text
features.parquet
psd_band_results.json
ppc_band_results.json
metadata.json
```

The uploaded archive contains source code, pretrained models and previous runs, but the default `computed_features/dethamp/raw_data` folder is not included in the visible repository root. To run new experiments, make sure the computed feature dataset is available at the expected path, or modify `ExperimentConfig.dataset_folder` and `ExperimentConfig.dataset_name`.

---

## Core concepts

### Macro sample vs micro-segment

A participant-level EEG recording is treated as a macro sample. For neural training, each macro EEG is split into several micro-segments. The neural backbone predicts one logit per micro-segment, then micro-level logits are aggregated into one macro-level Alzheimer probability.

Default micro-segmentation parameters:

```python
NeuroSymbolicEEGDataLoaderParameters(
    preprocessing_mode="mtdnet",
    n_micro_segments=60,
    batch_size=8,
)
```

A dataloader sample has the following structure:

```python
micro_x_raws, macro_x_feat, y_true = batch
```

where:

- `micro_x_raws` contains raw EEG micro-segments, usually shaped like `[batch, n_micro_segments, n_channels, n_times]`;
- `macro_x_feat` contains tabular EEG features used by symbolic rules;
- `y_true` contains the binary target.

### Supervised loss and logic loss

The trainer computes:

```text
total_loss = (1 - lambda_logic) * supervised_loss
           + lambda_logic * normalized_logic_loss
```

where:

- `supervised_loss` trains the neural network to classify AD vs CN;
- `logic_loss` penalizes predictions that violate extracted differentiable rules;
- `lambda_logic` controls the classification/logic trade-off.

`lambda_logic = 0.0` means pure neural supervised training. Larger values force the model to follow symbolic rules more strongly.

---

## EEG data manipulation

The `eeg/` package defines memory-aware wrappers around MNE raw objects.

### Main classes

| Class | File | Role |
|---|---|---|
| `EEGData` | `eeg/data.py` | Abstract base wrapper around an MNE `Raw` object with lazy loading. |
| `EEGRecordedData` | `eeg/data.py` | Raw EEG recording associated with a participant. |
| `EEGProcessedData` | `eeg/data.py` | EEG object produced by a preprocessing pipeline. |
| `EEGRecordedDataProvider` | `eeg/data.py` | Utility for loading recorded EEG data, including BIDS-style data. |
| `EEGProcessedDataIO` | `eeg/data.py` | Save/load utilities for processed EEG data. |
| `SampledSignal` | `eeg/signal.py` | One-dimensional sampled channel signal. |
| `SignalAnalysisEngine` | `eeg/signal.py` | Computes statistics, spectral and wavelet analysis for one signal. |
| `SpectralBand` | `eeg/signal.py` | Represents a named frequency band. |

### Lazy loading pattern

`EEGData` objects are built to avoid loading every raw recording permanently in memory. Use `.loaded()` when you need direct access to the MNE object:

```python
from features.io import FeaturesDatasetIO

features_dataset = FeaturesDatasetIO.load(
    "computed_features/dethamp/raw_data"
)

eeg = features_dataset.eegs[0]

with eeg.loaded() as raw:
    print(raw.ch_names)
    print(raw.get_data().shape)
```

### Iterating over EEG channels

```python
for signal in eeg.iter_signals():
    print(signal.name)
    print(signal.sampling_frequency)
    print(signal.points.shape)
```

Each `signal` is a `SampledSignal` with:

```python
signal.points       # NumPy array
signal.time_axis    # time vector in seconds
signal.name         # channel name
```

### Plotting raw EEG

```python
eeg.plot()
```

Internally this calls MNE's `raw.plot()`.

---

## Preprocessing pipelines

The `preprocessing/` package defines reusable preprocessing steps that operate on MNE `Raw` objects.

### Main files

| File | Object | Purpose |
|---|---|---|
| `preprocessing/pipeline.py` | `PreprocessingPipeline` | Orchestrates a sequence of preprocessing steps. |
| `preprocessing/step/base.py` | `PreprocessingStep` | Base interface for all preprocessing steps. |
| `preprocessing/step/bandpass.py` | `BandpassFilterStep` | Band-pass filtering. |
| `preprocessing/step/crop.py` | `CropStep` | Temporal cropping. |
| `preprocessing/step/detrend.py` | `DetrendStep`, `LocalDetrendStep` | Global/local detrending. |
| `preprocessing/step/hampel.py` | `HampelFilterStep` | Hampel outlier filtering. |
| `preprocessing/step/asr.py` | `ASRStep` | Artifact Subspace Reconstruction. |
| `preprocessing/names.py` | `PipelineName` | Pipeline-name enum. |

### Example: define and apply a pipeline

```python
from preprocessing.pipeline import PreprocessingPipeline
from preprocessing.step.bandpass import BandpassFilterStep
from preprocessing.step.crop import CropStep
from preprocessing.step.detrend import DetrendStep

pipeline = PreprocessingPipeline(
    name="bandpass_crop_detrend",
    steps=[
        BandpassFilterStep(l_freq=0.5, h_freq=48.0),
        CropStep(tmin=0.0, tmax=120.0),
        DetrendStep(),
    ],
)

processed_eeg = pipeline.compute(recorded_eeg)
```

The pipeline:

1. loads the source EEG if needed;
2. optionally prepares all steps;
3. copies the source MNE `Raw` object;
4. applies each step sequentially;
5. returns an `EEGProcessedData` object.

### Pipeline description

```python
print(pipeline.describe())
```

This returns a serializable dictionary with the pipeline name and every step description.

---

## Feature extraction

The `features/` package computes EEG biomarkers from processed EEG signals.

### Feature extraction configuration

`FeatureExtractionConfig` controls frequency bands and feature parameters:

```python
from features.config import FeatureExtractionConfig

config = FeatureExtractionConfig(
    bands={
        "delta": (1.0, 4.0),
        "theta": (5.0, 8.0),
        "alpha": (9.0, 13.0),
        "beta": (14.0, 30.0),
        "gamma": (31.0, 48.0),
        "full": (0.5, 48.0),
    },
    wavelet="db1",
    wavelet_level=1,
    ppc_epoch_duration=2.0,
)
```

### Feature categories

Feature classes are registered by category. The main families are:

| Family | File | Examples |
|---|---|---|
| Temporal | `features/definitions/temporal.py` | variance, skewness, kurtosis, peak amplitude, zero crossing rate, crest factor |
| Spectral | `features/definitions/spectral.py` | spectral centroid, spectral spread, spectral rolloff, alpha/gamma dominant frequency |
| Power ratios | `features/definitions/power_ratios.py` | theta/beta ratio, theta/alpha ratio, gamma/alpha ratio, spectral power ratio |
| Entropy | `features/definitions/entropy.py` | sample entropy, approximate entropy, permutation entropy |
| Complexity | `features/definitions/complexity.py` | Higuchi fractal dimension, Katz fractal dimension, Lempel-Ziv complexity, Hjorth parameters |
| Wavelets | `features/definitions/wavelets.py` | wavelet energy, relative wavelet energy, wavelet-packet energy |

### Extract scalar EEG features

```python
from features.config import FeatureExtractionConfig
from features.factory import FeatureExtractionEngine

config = FeatureExtractionConfig()
engine = FeatureExtractionEngine(config)

feature_result = engine.extract(processed_eeg)
```

### Extract PSD band powers

```python
from features.factory import PSDBandExtractionEngine

psd_engine = PSDBandExtractionEngine(config)
psd_result = psd_engine.extract(processed_eeg)

print(psd_result.band_powers_by_signal)
```

### Extract PPC connectivity matrices

```python
from features.factory import PPCBandExtractionEngine

ppc_engine = PPCBandExtractionEngine(config)
ppc_result = ppc_engine.extract(processed_eeg)

alpha_matrix = ppc_result.matrices_by_band["alpha"]
```

### Extract everything at once

```python
from features.factory import CompleteFeatureExtractionEngine

engine = CompleteFeatureExtractionEngine(config)
complete = engine.extract(processed_eeg)

features = complete.feature_result
psd = complete.psd_result
ppc = complete.ppc_result
```

This is more efficient than running all engines separately because scalar features and PSD band powers can reuse the same signal analysis.

---

## Feature datasets

The project represents computed features using participant-level and global dataset objects.

### Main classes

| Class | File | Role |
|---|---|---|
| `SingleParticipantProcessedFeatureDataset` | `features/dataset/participant.py` | Features, PSD, PPC and metadata for one participant. |
| `FeaturesDataset` | `features/dataset/base.py` | Global dataset containing all participants. |
| `SelectedFeaturesDataset` | `features/dataset/selected.py` | Restricted view keeping only selected feature families/columns. |
| `FeaturesDatasetSelector` | `features/dataset/selected.py` | High-level selection utility. |
| `SampleSelector` | `features/dataset/selector.py` | Row/sample filtering by metadata. |
| `FeaturesDatasetIO` | `features/io.py` | Load/export full feature datasets. |

### Load a dataset

```python
from features.io import FeaturesDatasetIO

features_dataset = FeaturesDatasetIO.load(
    "computed_features/dethamp/raw_data"
)
```

### Filter by health state

The training code keeps only Alzheimer and cognitively normal controls:

```python
dataset = features_dataset.selector.filter_by_healthstate(["AD", "CN"])
```

### Select feature families

The default experiment keeps these feature families:

```python
from features.dataset import FeaturesDatasetSelector

selected_dataset = FeaturesDatasetSelector.select(
    dataset,
    feature_family_names=[
        "theta_alpha_ratio",
        "spectral_power_ratio",
        "alpha",
        "beta",
        "gamma",
    ],
)
```

### Access machine-learning matrices

```python
X = selected_dataset.X
Y = selected_dataset.y
feature_names = selected_dataset.all_feature_names
subjects = selected_dataset.subject_dataframe
```

`X` is the explanatory matrix. `y` is derived from `subject_health`. `subject_dataframe` contains metadata such as subject ID, health state, group, gender, MMSE and age.

### Wide-column naming convention

Scalar channel features use:

```text
<channel>_<feature_name>
```

Example:

```text
Fp1_theta_alpha_ratio
Cz_spectral_rolloff
```

Connectivity features use:

```text
cn_<band>_<seed_channel>_<target_channel>
```

Example:

```text
cn_alpha_Fp1_Fp2
```

Subject-level features include:

```text
subject_group
subject_gender
subject_mmse
subject_age
```

---

## Statistical analysis

The `stats/` package provides a query-based statistical analysis layer.

### Main components

| Module | Purpose |
|---|---|
| `stats/queries/*` | Defines what should be tested: group comparison, correlation, factorial analysis. |
| `stats/engines/*` | Implements Wilcoxon rank-sum, t-test, Spearman, ANOVA, Tukey HSD. |
| `stats/correction/fdr.py` | Benjamini-Hochberg FDR correction. |
| `stats/bundles.py` | Extracts the correct samples from a `FeaturesDataset`. |
| `stats/results.py` | Result objects for scalar, pairwise and corrected tests. |
| `stats/runner.py` | Orchestrates queries, engines and correction. |

### Available test engines

- `WilcoxonRankSumEngine`: non-parametric group comparison;
- `TTestEngine`: parametric two-group comparison;
- `SpearmanEngine`: rank correlation, used for MMSE correlations;
- `OneWayANOVAEngine`, `TwoWayANOVAEngine`: group/factorial analyses;
- `TukeyHSDPostHocEngine`: post-hoc pairwise comparisons.

### Example: compare AD vs CN for one EEG feature

```python
from features.io import FeaturesDatasetIO
from stats.queries.factory import QueryFactory, QueryFactoryConfig
from stats.runner import StatisticalTestRunner

features_dataset = FeaturesDatasetIO.load("computed_features/dethamp/raw_data")

factory = QueryFactory(QueryFactoryConfig())
query = factory.compare_groups(
    variable="theta_alpha_ratio",
    group_col="subject_health",
    group_a="AD",
    group_b="CN",
    channel="Cz",
)

runner = StatisticalTestRunner(features_dataset)
outcome = runner.run([query])

print(outcome)
```

### Example: Spearman correlation with MMSE

```python
query = factory.correlate(
    x="theta_alpha_ratio",
    y="subject_mmse",
    channel="Cz",
)

outcome = runner.run([query])
```

### Example: apply FDR correction

```python
from stats.queries.specs import CorrectionSpec

query = factory.compare_groups(
    variable="alpha",
    group_col="subject_health",
    group_a="AD",
    group_b="CN",
    channel="Pz",
    correction=CorrectionSpec(method="fdr_bh"),
)
```

Use this layer to reproduce analyses such as:

1. Wilcoxon rank-sum tests between AD and CN;
2. FDR correction across many EEG features/channels;
3. Spearman correlation between EEG biomarkers and MMSE;
4. ANOVA/factorial tests when more than one factor is used.

---

## Decision trees and symbolic rules

The `prediction/decision_tree/` package trains interpretable decision trees on tabular EEG features.

### Main files

| File | Purpose |
|---|---|
| `base.py` | Simple wrapper around `sklearn.tree.DecisionTreeClassifier`. |
| `score.py` | Accuracy, balanced accuracy, F1 and confusion-matrix scoring. |
| `tunning.py` | Hyperparameter search for decision trees. |
| `feature_selection.py` | Forward feature selection using decision-tree performance. |
| `analysis.py` | Extracts node-level probabilities, split scores, leaf rules and feature importance. |
| `plot.py` | Decision-tree visualization. |

### Train a decision tree

```python
from prediction.decision_tree.base import DecisionTree, DecisionTreeParameters

params = DecisionTreeParameters(
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=10,
)

tree = DecisionTree(params)
trained_tree = tree.train(selected_dataset)
```

### Inspect the trained classifier

```python
clf = trained_tree.classifier
print(clf.get_depth())
print(clf.get_n_leaves())
```

### Why decision trees are used here

The decision tree is not the final predictive model. Its role is to extract human-readable rules from EEG biomarkers. These rules are then converted into differentiable rules and injected into the neural-network loss.

A typical rule has the form:

```text
IF Cz_theta_alpha_ratio <= threshold
AND Pz_alpha > threshold
THEN class = AD
```

---

## Differentiable rules

The `rules/` package transforms hard decision-tree rules into differentiable constraints.

### Main classes

| Class | File | Role |
|---|---|---|
| `Condition`, `DecisionRule` | `rules/decision_rule.py` | Symbolic hard rule representation. |
| `DifferentiableCondition` | `rules/differentiable_rule.py` | Soft version of a threshold condition. |
| `DifferentiableDecisionRule` | `rules/differentiable_rule.py` | Soft rule with differentiable truth degree. |
| `TruthDegreeEngine` | `rules/differentiable_rule.py` | Computes rule activation/truth degree. |
| `DifferentiableDecisionRulesFactory` | `rules/differentiable_rule.py` | Builds differentiable rules from a trained tree. |
| `TemperatureFeatureMappingFactory` | `rules/temperature.py` | Assigns soft-threshold temperature parameters. |
| `NeuroSymbolicRulesExtractor` | `rules/factory.py` | Full extraction pipeline from dataset to differentiable rules. |

### Extract differentiable rules

The training runner does this through the dataloader factory, but the conceptual flow is:

```python
from rules.factory import NeuroSymbolicRulesExtractorDefaultBuilder

extractor = NeuroSymbolicRulesExtractorDefaultBuilder.build(
    default_decision_tree=tree,
    random_seed=42,
    test_size=0.2,
    val_size=0.3,
)

rules, train_dataset, val_dataset, test_dataset = extractor.extract(selected_dataset)
```

Rules are sorted by score, and the experiment keeps only the strongest rules:

```python
rules = rules[:2]
```

The default number is controlled by:

```python
ExperimentConfig(n_rules_to_keep=2)
```

### Rule truth degree

A hard condition such as:

```text
feature <= threshold
```

is replaced by a smooth condition controlled by a temperature. This allows gradient-based training because the rule violation can be differentiated.

---

## Neural-network datasets and micro-segmentation

The `prediction/neural_network/dataset.py` file converts selected feature datasets into PyTorch dataloaders.

### Main classes

| Class | Role |
|---|---|
| `MTDNetSubjectSplitEngine` | Implements a fixed subject-independent split for the Miltiadous HC-AD task. |
| `EEGAugmentationParameters` | Controls time flip, channel shuffle and time masking. |
| `NeuroSymbolicEEGDataLoaderParameters` | Central configuration for PyTorch datasets and dataloaders. |
| `NeuroSymbolicEEGDataset` | Returns `(micro_x_raws, macro_x_feat, y_true)`. |
| `NeuroSymbolicEEGDataloaderFactory` | Builds train/val/test dataloaders and extracts rules. |

### Split strategies

Two split strategies are supported:

#### 1. `random`

Uses grouped random train/validation/test splitting controlled by:

```python
test_size=0.2
val_size=0.3
random_seed=42
```

This is the default in `ExperimentConfig` and in the CLI scripts.

#### 2. `mtdnet`

Uses a fixed subject-independent split for the Miltiadous HC-AD dataset:

```text
Train: sub-001 to sub-021 and sub-037 to sub-053
Val:   sub-022 to sub-028 and sub-054 to sub-059
Test:  sub-029 to sub-036 and sub-060 to sub-065
```

Use this when you want strict comparability with the MTDNet subject split.

### Build dataloaders manually

```python
from prediction.decision_tree.base import DecisionTree, DecisionTreeParameters
from prediction.neural_network.dataset import (
    NeuroSymbolicEEGDataLoaderParameters,
    NeuroSymbolicEEGDataloaderFactory,
)

params = NeuroSymbolicEEGDataLoaderParameters(
    batch_size=8,
    preprocessing_mode="mtdnet",
    split_strategy="random",
    random_seed=42,
    test_size=0.2,
    val_size=0.3,
    decision_tree=DecisionTree(
        DecisionTreeParameters(
            criterion="gini",
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=10,
        )
    ),
)

rules, train_loader, val_loader, test_loader = NeuroSymbolicEEGDataloaderFactory.build_all(
    features_dataset=selected_dataset,
    params=params,
)
```

---

## Neural backbones

The neural-network backbones live in `prediction/neural_network/neural_backbone/model.py`.

### `DeepEEGNet`

A simple 1D CNN:

```text
Conv1D -> BatchNorm -> ReLU -> Dropout
Conv1D -> BatchNorm -> ReLU -> Dropout
Conv1D -> BatchNorm -> ReLU -> AdaptiveAvgPool
Linear -> logit
```

### `MultiScaleDeepEEGNet`

The main model used by the experiment runner. It combines:

1. three temporal convolution branches with different kernel/stride scales;
2. concatenation of multi-scale features;
3. an LSTM over temporal blocks;
4. a small post-LSTM classifier.

Input shape:

```text
[batch_size, n_channels, n_times]
```

Output:

```text
[batch_size, 1]
```

Important architecture constraint:

```text
n_times must be divisible by 10
```

If not, `MultiScaleDeepEEGNet.forward()` raises a `ValueError`.

### Initialize model weights

```python
from prediction.neural_network.neural_backbone.model import MultiScaleDeepEEGNet
from prediction.neural_network.weight_init import EEGWeightInitializer

model = MultiScaleDeepEEGNet()
model = EEGWeightInitializer.apply(model, method="kaiming")
```

---

## Neuro-symbolic training

Training is implemented in `prediction/neural_network/neuro_symbolic/trainer.py`.

### Main trainer parameters

```python
from prediction.neural_network.neuro_symbolic.trainer import NeuroSymbolicDeepEEGTrainerParameters

params = NeuroSymbolicDeepEEGTrainerParameters(
    epochs=50,
    lr=1e-3,
    weight_decay=1e-4,
    lambda_logic=0.5,
    macro_aggregation_method="mean_probability",
    supervised_loss_compute_method="micro_bce",
    threshold=0.5,
    tensorboard_log_dir="runs/eeg_neurosymbolic/tensorboard/my_run",
)
```

### Macro aggregation

Micro-segment logits must be converted into one macro-level prediction. The code supports aggregation through `MicroLogitsToMacroProbabilityAggregator` and `MicroLogitsSupervisedLossAggregator`.

The default experiment uses:

```python
macro_aggregation_method="mean_probability"
supervised_loss_compute_method="micro_bce"
```

### Metrics

The trainer reports test metrics such as:

```text
test_total_loss
test_normalized_logic_loss
test_balanced_accuracy
test_f1_score
```

Balanced accuracy is especially important because EEG clinical datasets are often imbalanced.

---

## Command-line training

The from-scratch training script is:

```text
training/train.py
```

It runs several seeds and several `lambda_logic` values, then writes a comparison CSV.

### Basic command

```bash
PYTHONPATH=. python training/train.py
```

Default behavior:

```text
epochs = 50
split_strategy = random
seeds = 42 43 44 45 46
lambda_logic_values = 0.0 0.1 0.3 0.5
```

### Full syntax

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.0 0.1 0.3 0.5
```

### Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--epochs` | int | `50` | Number of training epochs per run. |
| `--split-strategy` | str | `random` | Dataset split strategy. Choices: `random`, `mtdnet`. |
| `--seeds` | int list | `42 43 44 45 46` | Seeds used for repeated experiments. |
| `--lambda-logic-values` | float list | `0.0 0.1 0.3 0.5` | Values of the neuro-symbolic loss weight to compare. |

### Run only one experiment

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 \
  --lambda-logic-values 0.5
```

### Compare many logic weights

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

### Use the MTDNet split

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy mtdnet \
  --seeds 42 \
  --lambda-logic-values 0.0 0.5
```

### Output of training

For each seed/lambda pair, the runner creates:

```text
runs/eeg_neurosymbolic_split_<split>_lambda_<lambda>_seed_<seed>/
├── checkpoints/
│   └── model.pt
├── logs/
│   └── training.log
└── outputs/
    └── history.npy
```

The global comparison file is:

```text
runs/comparison/lambda_logic_comparison.csv
```

This CSV contains one row per seed/lambda pair and includes all reported test metrics.

---

## Command-line fine-tuning

The fine-tuning script is:

```text
finetuning/finetune.py
```

It loads pretrained weights from `pretrained_models/model_<seed>.pt`, then fine-tunes them with a new `lambda_logic`, learning rate and weight decay.

### Basic command

```bash
PYTHONPATH=. python finetuning/finetune.py
```

Default behavior:

```text
epochs = 30
lr = 1e-4
weight_decay = 1e-5
split_strategy = random
seeds = 42 43 44 45 46
lambda_logic_values = 0.6 0.7 0.8 1.0
pretrained_model_dir = pretrained_models
```

### Full syntax

```bash
PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.6 0.7 0.8 1.0 \
  --pretrained-model-dir pretrained_models
```

### Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--epochs` | int | `30` | Number of additional fine-tuning epochs. |
| `--lr` | float | `1e-4` | Fine-tuning learning rate. |
| `--weight-decay` | float | `1e-5` | Fine-tuning weight decay. |
| `--split-strategy` | str | `random` | Dataset split strategy. Choices: `random`, `mtdnet`. |
| `--seeds` | int list | `42 43 44 45 46` | Seeds to fine-tune. Each seed expects a matching pretrained model. |
| `--lambda-logic-values` | float list | `0.6 0.7 0.8 1.0` | Logic weights used during fine-tuning. |
| `--pretrained-model-dir` | str | `pretrained_models` | Folder containing `model_<seed>.pt` files. |

### Run one fine-tuning experiment

```bash
PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --split-strategy random \
  --seeds 42 \
  --lambda-logic-values 0.7 \
  --pretrained-model-dir pretrained_models
```

This expects:

```text
pretrained_models/model_42.pt
```

### Compare several fine-tuning learning rates

Run the script multiple times:

```bash
PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.6 0.7 0.8

PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 5e-5 \
  --weight-decay 1e-5 \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.6 0.7 0.8
```

### Output of fine-tuning

For each run:

```text
runs/finetuning/eeg_neurosymbolic_finetune_split_<split>_lambda_<lambda>_lr_<lr>_wd_<weight_decay>_seed_<seed>/
├── checkpoints/
│   └── model.pt
├── logs/
│   └── training.log
└── outputs/
    └── history.npy
```

The comparison CSV is:

```text
runs/finetuning_comparison/finetuning_results.csv
```

---

## Command-line rule-following evaluation

The rule-following evaluation script is:

```text
evaluation/evaluate_rule_following.py
```

It loads saved models, rebuilds the same feature dataset and decision-tree rules for each seed, then computes:

- `rule_agreement`: how often the model follows active rules after thresholding;
- `rule_compliance`: soft weighted agreement between rule truth degree and class probability;
- `active_count`: weighted number of active rule cases;
- `truth_mass`: weighted soft activation mass;
- `n_rules`: number of evaluated rules.

### Required model filename pattern

The script expects model filenames with this exact pattern:

```text
model_seed-<seed>_lambda-<lambda>.pt
```

Examples:

```text
model_seed-42_lambda-0.6.pt
model_seed-43_lambda-0.8.pt
model_seed-46_lambda-1.0.pt
```

This is different from the training output path, where checkpoints are saved as `checkpoints/model.pt`. If you want to evaluate trained models with this script, copy or rename them into the expected pattern.

### Basic command

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics.csv
```

### Full syntax

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics.csv \
  --rule-activation-threshold 0.8 \
  --prediction-threshold 0.5 \
  --n-rules-to-keep 2 \
  --batch-size 8
```

### Arguments

| Argument | Type | Default | Description |
|---|---:|---:|---|
| `--models-dir` | str | required | Folder containing model files named `model_seed-*_lambda-*.pt`. |
| `--output-csv` | str | `rule_evaluation_results.csv` | Destination CSV path. |
| `--rule-activation-threshold` | float | `0.8` | Minimum truth degree for a rule to be considered active in hard agreement. |
| `--prediction-threshold` | float | `0.5` | Probability threshold used to convert model probability into a class decision. |
| `--n-rules-to-keep` | int | `2` | Number of strongest rules evaluated per seed. |
| `--batch-size` | int | `8` | Evaluation dataloader batch size. |

### Example: evaluate all models in `evaluation/models`

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics.csv \
  --rule-activation-threshold 0.8 \
  --prediction-threshold 0.5 \
  --n-rules-to-keep 2
```

### Example: stricter rule activation

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics_strict.csv \
  --rule-activation-threshold 0.9 \
  --prediction-threshold 0.5
```

### Example: evaluate more rules

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics_5_rules.csv \
  --n-rules-to-keep 5
```

---

## Outputs and experiment folders

### From-scratch training folders

```text
runs/eeg_neurosymbolic_split_random_lambda_0.5_seed_42/
├── checkpoints/model.pt
├── logs/training.log
└── outputs/history.npy
```

### Fine-tuning folders

```text
runs/finetuning/eeg_neurosymbolic_finetune_split_random_lambda_0.7_lr_0.0001_wd_1e-05_seed_42/
├── checkpoints/model.pt
├── logs/training.log
└── outputs/history.npy
```

### Comparison CSVs

```text
runs/comparison/lambda_logic_comparison.csv
runs/finetuning_comparison/finetuning_results.csv
evaluation/results/rule_metrics.csv
```

### History files

`history.npy` contains the training history returned by the trainer. Load it with:

```python
import numpy as np

history = np.load(
    "runs/eeg_neurosymbolic_split_random_lambda_0.5_seed_42/outputs/history.npy",
    allow_pickle=True,
)
print(history)
```

---

## TensorBoard

Training and fine-tuning use TensorBoard through `SummaryWriter`.

### From-scratch training logs

The trainer uses:

```text
runs/eeg_neurosymbolic/tensorboard/<run_name>
```

Start TensorBoard with:

```bash
tensorboard --logdir runs/eeg_neurosymbolic/tensorboard
```

### Fine-tuning logs

Fine-tuning uses:

```text
runs/finetuning/tensorboard/<run_name>
```

Start TensorBoard with:

```bash
tensorboard --logdir runs/finetuning/tensorboard
```

Then open the local TensorBoard URL printed in the terminal.

---

## Typical experiment workflows

### Workflow 1: baseline neural training without logic

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.0
```

Use this to measure pure supervised neural performance.

### Workflow 2: neuro-symbolic lambda sweep

```bash
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

Use this to study the trade-off between balanced accuracy and rule compliance.

### Workflow 3: fine-tune pretrained neural models with stronger logic

```bash
PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.6 0.7 0.8 1.0 \
  --pretrained-model-dir pretrained_models
```

Use this when you already have good pretrained models and want to increase logical consistency without restarting from scratch.

### Workflow 4: evaluate rule following

First ensure models follow the naming convention:

```text
evaluation/models/model_seed-42_lambda-0.6.pt
```

Then run:

```bash
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics.csv \
  --rule-activation-threshold 0.8 \
  --prediction-threshold 0.5 \
  --n-rules-to-keep 2
```

### Workflow 5: merge performance and rule metrics

After training/fine-tuning and rule evaluation, merge CSVs using pandas:

```python
import pandas as pd

performance = pd.read_csv("runs/finetuning_comparison/finetuning_results.csv")
rules = pd.read_csv("evaluation/results/rule_metrics.csv")

merged = performance.merge(
    rules,
    on=["seed", "lambda_logic"],
    how="inner",
)

merged.to_csv("analysis/merged_finetuning_rule_metrics.csv", index=False)
print(merged.head())
```

This is useful for plotting:

```text
rule_compliance = f(test_balanced_accuracy)
rule_agreement = f(lambda_logic)
test_balanced_accuracy = f(lambda_logic)
```

---

## Extending the project

### Add a new EEG feature

1. Create a feature class in the appropriate file under `features/definitions/`.
2. Inherit from `EEGFeature` or follow the existing feature pattern.
3. Register the feature with `@register_feature`.
4. Assign it a `FeatureCategory`.
5. Recompute feature datasets.
6. Select it by name through `FeaturesDatasetSelector.select()`.

Example conceptual pattern:

```python
from features.definitions.base import EEGFeature, register_feature
from features.categories import FeatureCategory

@register_feature
class MyNewFeature(EEGFeature):
    name = "my_new_feature"
    category = FeatureCategory.SPECTRAL

    @staticmethod
    def compute(context):
        value = ...
        return value
```

### Change selected features for training

Edit `training/config.py`:

```python
feature_family_names: tuple[str, ...] = (
    "theta_alpha_ratio",
    "theta_beta_ratio",
    "gamma_alpha_ratio",
    "spectral_rolloff",
    "alpha_dominant_frequency",
    "gamma_dominant_frequency",
)
```

Then rerun training:

```bash
PYTHONPATH=. python training/train.py --seeds 42 --lambda-logic-values 0.5
```

### Change decision-tree hyperparameters

Edit `EEGExperimentRunner.build_decision_tree()` in `training/experiment.py`:

```python
DecisionTreeParameters(
    criterion="gini",
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=10,
)
```

Typical robust alternatives:

```python
DecisionTreeParameters(
    criterion="entropy",
    max_depth=3,
    min_samples_split=10,
    min_samples_leaf=15,
)
```

Larger trees may improve training accuracy but produce less stable and less interpretable rules.

### Change the neural model

Edit `EEGExperimentRunner.build_model()`:

```python
from prediction.neural_network.neural_backbone.model import DeepEEGNet

model = DeepEEGNet(in_channels=19)
```

or customize `MultiScaleDeepEEGNet`:

```python
model = MultiScaleDeepEEGNet(
    in_nch=19,
    first_layer_ch=32,
    lstm_nch=16,
    post_lin_weights=[16],
    out_nch=1,
)
```

### Change loss behavior

Edit `ExperimentConfig`:

```python
macro_aggregation_method: str = "mean_probability"
supervised_loss_compute_method: str = "micro_bce"
```

Edit trainer normalization parameters if needed:

```python
loss_scale_alpha = 0.99
loss_scale_eps = 1e-8
```

### Add a new command-line argument

For example, to expose learning rate in `training/train.py`:

1. Add an argument in `parse_args()`:

```python
parser.add_argument("--lr", type=float, default=1e-3)
```

2. Pass it to `ExperimentConfig`:

```python
config = ExperimentConfig(
    lambda_logic=lambda_logic,
    epochs=args.epochs,
    random_seed=seed,
    split_strategy=args.split_strategy,
    lr=args.lr,
)
```

3. Run:

```bash
PYTHONPATH=. python training/train.py --lr 5e-4
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'training'`

Run commands from the repository root and prefix them with `PYTHONPATH=.`:

```bash
PYTHONPATH=. python training/train.py
```

### `FileNotFoundError: computed_features/dethamp/raw_data`

The computed feature dataset is missing. Either place the dataset at:

```text
computed_features/dethamp/raw_data
```

or modify:

```python
ExperimentConfig(dataset_folder="...", dataset_name="...")
```

### `Pretrained model not found: pretrained_models/model_<seed>.pt`

Fine-tuning expects one pretrained file per seed:

```text
pretrained_models/model_42.pt
pretrained_models/model_43.pt
...
```

Either add the missing file or restrict `--seeds`:

```bash
PYTHONPATH=. python finetuning/finetune.py --seeds 42
```

### `No model found with pattern model_seed-*_lambda-*.pt`

The rule-following evaluator requires files named like:

```text
model_seed-42_lambda-0.6.pt
```

Training saves models as:

```text
runs/.../checkpoints/model.pt
```

Copy/rename them before evaluation.

Example:

```bash
mkdir -p evaluation/models
cp runs/eeg_neurosymbolic_split_random_lambda_0.6_seed_42/checkpoints/model.pt \
   evaluation/models/model_seed-42_lambda-0.6.pt
```

### `n_times must be divisible by 10`

`MultiScaleDeepEEGNet` requires input EEG segment length divisible by 10. Adjust preprocessing, cropping or micro-segmentation so each segment length satisfies this constraint.

### CUDA not used

The runner automatically uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is not detected, check your PyTorch installation:

```bash
python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.__version__)
PY
```

### Rule compliance does not reach 100%

This is expected. `rule_compliance` is a soft probability-weighted metric. A perfect value would imply that the model outputs extreme probabilities fully aligned with active rules, which is generally unrealistic and may indicate overfitting or poor calibration. Moderate improvements in compliance can already be meaningful when balanced accuracy remains stable.

---

## Minimal reproducible commands

From the repository root:

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train from scratch
PYTHONPATH=. python training/train.py \
  --epochs 50 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.0 0.1 0.3 0.5

# Fine-tune pretrained models
PYTHONPATH=. python finetuning/finetune.py \
  --epochs 30 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --split-strategy random \
  --seeds 42 43 44 45 46 \
  --lambda-logic-values 0.6 0.7 0.8 1.0 \
  --pretrained-model-dir pretrained_models

# Evaluate rule following
PYTHONPATH=. python evaluation/evaluate_rule_following.py \
  --models-dir evaluation/models \
  --output-csv evaluation/results/rule_metrics.csv \
  --rule-activation-threshold 0.8 \
  --prediction-threshold 0.5 \
  --n-rules-to-keep 2 \
  --batch-size 8

# TensorBoard
tensorboard --logdir runs/eeg_neurosymbolic/tensorboard
```

---

## Project summary

This repository provides a complete neuro-symbolic EEG pipeline:

1. EEG recordings are represented as lazy MNE-based objects.
2. Preprocessing pipelines transform raw recordings into processed EEG objects.
3. Feature extraction computes scalar biomarkers, PSD band powers and PPC connectivity.
4. Feature datasets aggregate participant-level biomarkers into machine-learning matrices.
5. Statistical tools test group differences and clinical correlations.
6. Decision trees extract interpretable biomarker rules.
7. Differentiable rules transform symbolic thresholds into soft constraints.
8. A multi-scale CNN-LSTM predicts Alzheimer probability from EEG micro-segments.
9. Training combines supervised classification loss and normalized logic loss.
10. Evaluation measures both predictive performance and rule-following behavior.

The most important experimental parameter is `lambda_logic`: it controls the trade-off between raw classification performance and consistency with interpretable EEG-derived rules.
