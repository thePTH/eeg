from dataclasses import dataclass

@dataclass(frozen=True)
class DecisionTreeParameters:
    criterion:str="entropy"
    max_depth:int=5
    min_samples_leaf:int=5
    min_samples_split:int=10
    class_weight:str="balanced"
    random_state:int=42