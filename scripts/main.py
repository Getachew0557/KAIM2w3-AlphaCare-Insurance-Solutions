import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../notebook/model/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_loader import load_data, display_basic_info, print_unique_values
from visualization import plot_postalcode_premium, plot_premium_vs_claims
from evaluation import evaluate_model
from save_model import save_model
from feature_importance import plot_feature_importance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from eda import (
    check_missing_values, 
    plot_missing_values, 
    plot_histograms, 
    plot_categorical_bars, 
    plot_correlation_matrix, 
    plot_categorical_bars, 
    data_compression
)
from data_preprocessing import (
    handle_missing_values, 
    get_numerical_columns, 
    convert_datetime_features, 
    encode_categorical_features
)
from modeling import (
    train_linear_regression, 
    train_decision_tree, 
    train_random_forest, 
    train_xgboost
)
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
    
    # conver data  and encode features
    df = convert_datetime_features(df)
    df = encode_categorical_features(df)

    # Define features and target
    X = df[numerical_cols]
    y_premium = df['TotalPremium']
    y_claims = df['TotalClaims']

    # Train models
    models, X_train_premium, X_test_premium, y_train_premium, y_test_premium, X_train_claims, X_test_claims, y_train_claims, y_test_claims = train_all_models(X, y_premium, y_claims)

    # Save models
    save_model(models['premium_lr'], 'model_premium_lr.pkl')
    save_model(models['claims_lr'], 'model_claims_lr.pkl')
    save_model(models['premium_rf'], 'model_premium_rf.pkl')
    save_model(models['claims_rf'], 'model_claims_rf.pkl')
    save_model(models['premium_xgb'], 'model_premium_xgb.pkl')
    save_model(models['claims_xgb'], 'model_claims_xgb.pkl')
    save_model(models['premium_dt'], 'model_premium_dt.pkl')
    save_model(models['claims_dt'], 'model_claims_dt.pkl')

    # Evaluate models
    for model_name, model in models.items():
        if 'premium' in model_name:
            X_test, y_test = X_test_premium, y_test_premium
        else:
            X_test, y_test = X_test_claims, y_test_claims

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{model_name}: MSE={mse}, MAE={mae}, R^2={r2}")

    # Feature Importance
    for model_name in ['premium_rf', 'claims_rf', 'premium_xgb', 'claims_xgb']:
        if model_name in models:
            plot_feature_importance(models[model_name], X)
    # Feature Importance
    plot_feature_importance(model_premium_rf, X)
    plot_feature_importance(model_claims_rf, X)
    plot_feature_importance(model_premium_xgb, X)
    plot_feature_importance(model_claims_xgb, X)

if __name__ == "__main__":
    main()
