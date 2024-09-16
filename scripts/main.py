import os
import sys
import pandas as pd

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_loader import load_data, display_basic_info, print_unique_values
from eda import check_missing_values, plot_missing_values, plot_histograms, plot_categorical_bars, plot_correlation_matrix, plot_categorical_bars, data_compression
from data_preprocessing import handle_missing_values, get_numerical_columns
from visualization import plot_postalcode_premium, plot_premium_vs_claims
from hypothesis_testing import (
    test_province_risk_difference, 
    test_zip_code_risk_difference, 
    calculate_margin, 
    test_margin_difference_by_zip, 
    test_gender_risk_difference
)

def main():
    # Data Collection
    file_path = '../data/MachineLearningRating_v3.txt'
    df = load_data(file_path)
    
    # Display basic information about the dataset
    display_basic_info(df)
    print_unique_values(df)

    # EDA
    check_missing_values(df)
    plot_missing_values(df)
    plot_histograms(df)
    plot_categorical_bars(df)
    plot_correlation_matrix(df)

    # Data Preprocessing
    df = handle_missing_values(df)
    numerical_cols = get_numerical_columns(df)
    print("Updated Numerical Columns List:")
    print(numerical_cols)

    data_compression(df)

    # Visualizations
    plot_postalcode_premium(df)
    plot_premium_vs_claims(df)

        # Test risk differences across provinces
    province_results = test_province_risk_difference(df)
    print("Risk Differences Across Provinces:")
    for result in province_results:
        province1, province2, t_stat, p_value, reject_null = result
        print(f"Province {province1} vs {province2}: T-stat={t_stat}, P-value={p_value}, Reject Null: {reject_null}")

    # Test risk differences between zip codes
    zip_code_results = test_zip_code_risk_difference(df)
    print("\nRisk Differences Between Zip Codes:")
    for result in zip_code_results:
        zip1, zip2, t_stat, p_value, reject_null = result
        print(f"Zip {zip1} vs {zip2}: T-stat={t_stat}, P-value={p_value}, Reject Null: {reject_null}")

    # Calculate margin and test margin differences between zip codes
    df = calculate_margin(df)
    margin_results = test_margin_difference_by_zip(df)
    print("\nMargin Differences Between Zip Codes:")
    for result in margin_results:
        zip1, zip2, t_stat, p_value, reject_null = result
        print(f"Zip {zip1} vs {zip2}: T-stat={t_stat}, P-value={p_value}, Reject Null: {reject_null}")

    # Test risk differences between Men and Women
    t_stat, p_value, reject_null = test_gender_risk_difference(df)
    print("\nRisk Differences Between Men and Women:")
    print(f"T-stat={t_stat}, P-value={p_value}, Reject Null: {reject_null}")

if __name__ == "__main__":
    main()
