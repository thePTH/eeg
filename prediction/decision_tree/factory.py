from prediction.decision_tree.hyperparameters import DecisionTreeParameters
from features.dataset import FeaturesDataset
from sklearn.tree import DecisionTreeClassifier
from prediction.decision_tree.predictor import TrainedDecisionTreePredictor


class DecisionTreeFactory:

    @staticmethod
    def build(train_dataset:FeaturesDataset, parameters:DecisionTreeParameters=DecisionTreeParameters()):
        X_train, y_train = train_dataset.selector.Xy()
        
        
        decision_tree = DecisionTreeClassifier(
            criterion=parameters.criterion,
            max_depth=parameters.max_depth,
            min_samples_leaf=parameters.min_samples_leaf,
            min_samples_split=parameters.min_samples_split,
            class_weight=parameters.class_weight,
            random_state=parameters.random_state
        )

        decision_tree.fit(X_train, y_train)

        return TrainedDecisionTreePredictor(decision_tree, train_dataset, parameters)