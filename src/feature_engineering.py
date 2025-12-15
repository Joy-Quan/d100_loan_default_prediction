import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class CustomStandardScaler(BaseEstimator, TransformerMixin):
    """
    A custom implementation of StandardScaler.
    It standardizes features by removing the mean and scaling to unit variance.
    z = (x - u) / s
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        """
        Compute the mean and std to be used for later scaling.
        """
        # Convert to numpy array for calculation
        X = np.array(X)
        
        # Calculate mean and standard deviation along axis 0 (columns)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        
        # Handle constant features (std=0) to avoid division by zero
        # If std is 0, we set it to 1, so the value becomes (x - mean) / 1 = 0
        self.scale_[self.scale_ == 0] = 1.0
        
        self.n_features_in_ = X.shape[1]
        
        return self

    def transform(self, X):
        """
        Perform standardization by centering and scaling.
        """
        check_is_fitted(self, ['mean_', 'scale_'])
        
        X = np.array(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Shape of input {X.shape} does not match the fitted shape.")
            
        # The core logic: (X - mean) / std
        X_scaled = (X - self.mean_) / self.scale_
        
        return X_scaled

    def get_feature_names_out(self, input_features=None):
        """
        Returns feature names for output (needed for Pipeline compatibility).
        """
        if input_features is None:
            return [f"x{i}" for i in range(self.n_features_in_)]
        return input_features
