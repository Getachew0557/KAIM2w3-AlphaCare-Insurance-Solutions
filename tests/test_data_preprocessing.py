import os
import sys
import pytest
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_preprocessing import handle_missing_values, get_numerical_columns
# Sample Data
data = {
    'NumberOfVehiclesInFleet': [10, None, 5, 2],
    'CrossBorder': [None, 'No', 'Yes', 'No'],
    'CustomValueEstimate': [5000, 3000, None, 10000],
    'NewVehicle': ['Yes', 'No', 'Yes', None],
    'TotalPremium': [2000, 1500, None, 2500],
    'TotalClaims': [1, 0, 2, None],
    'PostalCode': ['12345', '54321', '12345', '54321']
}
df = pd.DataFrame(data)

def test_handle_missing_values():
    # df_clean = handle_missing_values(df)
    
    # # Check if columns were dropped correctly
    # assert 'NumberOfVehiclesInFleet' not in df_clean.columns
    # assert 'CrossBorder' not in df_clean.columns
    # assert 'CustomValueEstimate' not in df_clean.columns
    
    # # Check if categorical and numerical columns are imputed
    # assert df_clean['NewVehicle'].notnull().all()
    # assert df_clean['TotalPremium'].notnull().all()
    pass

def test_get_numerical_columns():
    numerical_cols = get_numerical_columns(df)
    assert 'TotalPremium' in numerical_cols
    assert 'TotalClaims' in numerical_cols
    assert 'PostalCode' not in numerical_cols