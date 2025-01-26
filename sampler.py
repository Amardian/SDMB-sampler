import pandas as pd
import numpy as np
import random
from typing import Union

class SDMBSampler:
    """SDMB-Sampler: Statistical Density Minority Balancing Sampler for class imbalance"""
    
    def __init__(self, target_column: Union[str, int] = 10, 
                 features_to_change: Union[int, float, str] = 'auto',
                 random_state: int = None):
        self.target_column = target_column
        self.features_to_change = features_to_change
        self.random_state = random_state
        self.minority_stats = {}
        self.majority_stats = {}
        
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

    def _validate_input(self, data: pd.DataFrame):
        if self.target_column not in data.columns:
            raise ValueError(f"Target column {self.target_column} not found")
        if len(data[self.target_column].unique()) != 2:
            raise ValueError("Binary classification required")

    def _compute_stats(self, data: pd.DataFrame, class_val: int):
        class_data = data[data[self.target_column] == class_val]
        return {
            'means': class_data.mean().values,
            'stds': class_data.std().values,
            'mins': class_data.min().values,
            'maxs': class_data.max().values,
            'modes': class_data.mode().mean().values
        }

    def fit_resample(self, data: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(data)
        
        minority = data[data[self.target_column] == 1]
        majority = data[data[self.target_column] == 0]
        
        self.minority_stats = self._compute_stats(data, 1)
        self.majority_stats = self._compute_stats(data, 0)
        
        n_samples = len(majority) - len(minority)
        if n_samples <= 0:
            return data
        
        features = [col for col in data.columns if col != self.target_column]
        n_features = len(features)
        
        if self.features_to_change == 'auto':
            change_features = max(1, n_features//5)
        elif isinstance(self.features_to_change, float):
            change_features = max(1, int(n_features*self.features_to_change))
        else:
            change_features = min(n_features, self.features_to_change)
        
        synthetic = []
        for _ in range(n_samples):
            sample = minority.sample(1).iloc[0].copy()
            for feat_idx in random.sample(range(n_features), change_features):
                col = features[feat_idx]
                min_val = self.minority_stats['mins'][feat_idx]
                max_val = self.minority_stats['maxs'][feat_idx]
                new_val = np.clip(
                    random.uniform(min_val, max_val),
                    min_val,
                    max_val
                )
                sample[col] = new_val
            synthetic.append(sample)
            
        return pd.concat([majority, minority, pd.DataFrame(synthetic)])\
               .sample(frac=1, random_state=self.random_state)\
               .reset_index(drop=True)