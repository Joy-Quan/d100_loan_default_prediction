import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import CustomStandardScaler

# Define test cases: 
# Case 1: Simple 2D array
# Case 2: Data with a constant column (std=0)
# Case 3: Negative values
test_cases = [
    (np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])),
    (np.array([[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]])),  # First col is constant
    (np.array([[-10.0, -2.0], [0.0, 0.0], [10.0, 2.0]])),
]

@pytest.mark.parametrize("data", test_cases)
def test_custom_standard_scaler_vs_sklearn(data):
    """
    Test that our CustomStandardScaler matches the output of sklearn's StandardScaler.
    """
    # 1. Initialize both scalers
    my_scaler = CustomStandardScaler()
    sklearn_scaler = StandardScaler()

    # 2. Fit and transform
    my_output = my_scaler.fit_transform(data)
    sklearn_output = sklearn_scaler.fit_transform(data)

    # 3. Compare results
    # We use np.allclose to allow for tiny floating point differences
    assert np.allclose(my_output, sklearn_output), "Output does not match sklearn implementation!"
    
    # 4. Check attributes
    assert np.allclose(my_scaler.mean_, sklearn_scaler.mean_), "Means do not match!"
    assert np.allclose(my_scaler.scale_, sklearn_scaler.scale_), "Scales (std) do not match!"

def test_dataframe_support():
    """Test that it works with pandas DataFrame"""
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    scaler = CustomStandardScaler()
    output = scaler.fit_transform(df)
    
    assert isinstance(output, np.ndarray)
    assert output.shape == (3, 2)
