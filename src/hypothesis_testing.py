import pandas as pd
from scipy.stats import ttest_ind
from itertools import combinations

def test_province_risk_difference(df):
    """Test the risk difference across provinces."""
    provinces = df['Province'].unique()
    
    results = []
    for province1, province2 in combinations(provinces, 2):
        group1 = df[df['Province'] == province1]['TotalClaims']
        group2 = df[df['Province'] == province2]['TotalClaims']
        t_stat, p_value = ttest_ind(group1, group2)
        results.append((province1, province2, t_stat, p_value, p_value < 0.05))
    
    return results

def test_zip_code_risk_difference(df):
    """Test the risk difference between zip codes."""
    zip_codes = df['PostalCode'].unique()
    
    results = []
    for zip1, zip2 in combinations(zip_codes, 2):
        group1 = df[df['PostalCode'] == zip1]['TotalClaims']
        group2 = df[df['PostalCode'] == zip2]['TotalClaims']
        t_stat, p_value = ttest_ind(group1, group2)
        results.append((zip1, zip2, t_stat, p_value, p_value < 0.05))
    
    return results

def calculate_margin(df):
    """Calculate margin as TotalPremium - TotalClaims."""
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def test_margin_difference_by_zip(df):
    """Test the margin difference between zip codes."""
    zip_codes = df['PostalCode'].unique()
    
    results = []
    for zip1, zip2 in combinations(zip_codes, 2):
        margin1 = df[df['PostalCode'] == zip1]['Margin']
        margin2 = df[df['PostalCode'] == zip2]['Margin']
        t_stat, p_value = ttest_ind(margin1, margin2)
        results.append((zip1, zip2, t_stat, p_value, p_value < 0.05))
    
    return results

def test_gender_risk_difference(df):
    """Test the risk difference between Men and Women."""
    group_male = df[df['Gender'] == 'Male']['TotalClaims']
    group_female = df[df['Gender'] == 'Female']['TotalClaims']
    
    t_stat, p_value = ttest_ind(group_male, group_female)
    
    return t_stat, p_value, p_value < 0.05