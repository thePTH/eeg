from features.dataset import SelectedFeaturesDataset
from prediction.decision_tree.tunning import HyperparameterGrid, DecisionTreeScoreEngine,DecisionTreeOptimizer, DecisionTree
from prediction.decision_tree.feature_selection import DecisionTreeFeatureSelectionTrainer
from rules.decision_rule import DecisionRulesFactory
from rules.differentiable_rule import DifferentiableDecisionRulesFactory

from dataclasses import dataclass

@dataclass
class DecisionTreeScoreEngineParameters:
    n_splits: int = 5
    scoring: str = "balanced_accuracy"
    random_seed:int=42


@dataclass
class DecisionTreeOptimizerParameters:
    hyperparamter_grid:HyperparameterGrid=HyperparameterGrid()
    lambda_std:float=0
    default_optimized_decision_tree:DecisionTree=None


@dataclass
class DecisionTreeFeatureSelectionTrainerParameters:
    lambda_std :float=0
    random_seed :int= 42
    output_test_size:float=0.2
    output_val_size:float=0.3

@dataclass
class TemperatureFeatureMappingFactoryParameters:
    c_tau: float = 0.1
    min_tau: float = 0.001






class NeuroSymbolicRulesExtractor:

    def __init__(self, decision_tree_score_engine_params:DecisionTreeScoreEngineParameters, decision_tree_optimizer_params:DecisionTreeOptimizerParameters, decision_tree_trainer_params:DecisionTreeFeatureSelectionTrainerParameters, temperature_feature_mapping_params:TemperatureFeatureMappingFactoryParameters):
        self.decision_tree_score_engine_params = decision_tree_score_engine_params
        self.decision_tree_optimizer_params = decision_tree_optimizer_params
        self.decision_tree_trainer_params = decision_tree_trainer_params
        self.temperature_feature_mapping_params = temperature_feature_mapping_params





    
    def extract(self, dataset:SelectedFeaturesDataset):
        score_engine = DecisionTreeScoreEngine(n_splits=self.decision_tree_score_engine_params.n_splits, scoring=self.decision_tree_score_engine_params.scoring, random_seed=self.decision_tree_score_engine_params.random_seed)

        if self.decision_tree_optimizer_params.default_optimized_decision_tree :
            optimized_decision_tree = self.decision_tree_optimizer_params.default_optimized_decision_tree
        else:
            hyperparamter_grid = self.decision_tree_optimizer_params.hyperparamter_grid
            hyperparamter_searcher = DecisionTreeOptimizer(dataset=dataset, score_engine=score_engine)
            optimized_decision_tree, _ = hyperparamter_searcher.optimize(hyperparamter_grid, lambda_std = self.decision_tree_optimizer_params.lambda_std)

        decision_tree_trainer = DecisionTreeFeatureSelectionTrainer(score_engine=score_engine, lambda_std=self.decision_tree_trainer_params.lambda_std, random_seed=self.decision_tree_trainer_params.random_seed, output_test_size=self.decision_tree_trainer_params.output_test_size, output_val_size=self.decision_tree_trainer_params.output_val_size)
        trained_tree, val_dataset, test_dataset = decision_tree_trainer.train(optimized_decision_tree, dataset)
        train_dataset = trained_tree.dataset
        
        differentiable_decision_rules, _ = DifferentiableDecisionRulesFactory.build(trained_tree, c_tau=self.temperature_feature_mapping_params.c_tau, min_tau=self.temperature_feature_mapping_params.min_tau)
        rules = sorted(differentiable_decision_rules, key=lambda rule : rule.score, reverse=True)

        return rules, train_dataset,val_dataset, test_dataset






class NeuroSymbolicRulesExtractorDefaultBuilder:
    @staticmethod
    def build(default_decision_tree:DecisionTree, random_seed = 42, test_size = 0.2, val_size=0.3):
        decision_tree_score_engine_params = DecisionTreeScoreEngineParameters(random_seed=random_seed)
        decision_tree_optimizer_params = DecisionTreeOptimizerParameters(default_optimized_decision_tree=default_decision_tree)
        decision_tree_trainer_params = DecisionTreeFeatureSelectionTrainerParameters(random_seed=random_seed, output_test_size=test_size, output_val_size=val_size)
        temperature_feature_mapping_params = TemperatureFeatureMappingFactoryParameters()
        neuro_symbolic_rules_extractor = NeuroSymbolicRulesExtractor(decision_tree_score_engine_params, decision_tree_optimizer_params, decision_tree_trainer_params, temperature_feature_mapping_params)

        return neuro_symbolic_rules_extractor
