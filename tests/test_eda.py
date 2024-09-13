import os
import sys
import pytest
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from eda import check_missing_values, describe_data, plot_missing_values, plot_histograms, plot_categorical_bars, plot_correlation_matrix

# Sample Data
data = {
    'TotalPremium': [2000, 1500, 3000, 2500],
    'TotalClaims': [1, 0, 2, 1],
    'PostalCode': ['12345', '54321', '12345', '54321']
}
df = pd.DataFrame(data)

def test_check_missing_values():
    missing_values = check_missing_values(df)
    assert missing_values['TotalPremium'] == 0
    assert missing_values['TotalClaims'] == 0

def test_describe_data():
    description = describe_data(df)
    assert 'TotalPremium' in description.columns
    assert 'TotalClaims' in description.columns
