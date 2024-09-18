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
from evaluation import evaluate_model, plot_metrics
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
    split_data, 
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


    # Split data into train and test sets
    (X_train_premium, X_test_premium, y_train_premium, y_test_premium, 
     X_train_claims, X_test_claims, y_train_claims, y_test_claims) = split_data(X, y_premium, y_claims)

    # Train models
    model_premium_dt = train_decision_tree(X_train_premium, y_train_premium)
    model_claims_dt = train_decision_tree(X_train_claims, y_train_claims)

    model_premium_lr = train_linear_regression(X_train_premium, y_train_premium)
    model_claims_lr = train_linear_regression(X_train_claims, y_train_claims)

    model_premium_rf = train_random_forest(X_train_premium, y_train_premium)
    model_claims_rf = train_random_forest(X_train_claims, y_train_claims)

    model_premium_xgb = train_xgboost(X_train_premium, y_train_premium)
    model_claims_xgb = train_xgboost(X_train_claims, y_train_claims)

    # Evaluate models
    metrics_premium_dt = evaluate_model(model_premium_dt, X_test_premium, y_test_premium)
    metrics_claims_dt = evaluate_model(model_claims_dt, X_test_claims, y_test_claims)

    metrics_premium_lr = evaluate_model(model_premium_lr, X_test_premium, y_test_premium)
    metrics_claims_lr = evaluate_model(model_claims_lr, X_test_claims, y_test_claims)

    metrics_premium_rf = evaluate_model(model_premium_rf, X_test_premium, y_test_premium)
    metrics_claims_rf = evaluate_model(model_claims_rf, X_test_claims, y_test_claims)

    metrics_premium_xgb = evaluate_model(model_premium_xgb, X_test_premium, y_test_premium)
    metrics_claims_xgb = evaluate_model(model_claims_xgb, X_test_claims, y_test_claims)

    # Print the evaluation metrics
    print(f"Decision Tree - TotalPremium: MSE={metrics_premium_dt[0]}, MAE={metrics_premium_dt[1]}, R²={metrics_premium_dt[2]}")
    print(f"Decision Tree - TotalClaims: MSE={metrics_claims_dt[0]}, MAE={metrics_claims_dt[1]}, R²={metrics_claims_dt[2]}")
    print(f"Linear Regression - TotalPremium: MSE={metrics_premium_lr[0]}, MAE={metrics_premium_lr[1]}, R²={metrics_premium_lr[2]}")
    print(f"Linear Regression - TotalClaims: MSE={metrics_claims_lr[0]}, MAE={metrics_claims_lr[1]}, R²={metrics_claims_lr[2]}")
    print(f"Random Forest - TotalPremium: MSE={metrics_premium_rf[0]}, MAE={metrics_premium_rf[1]}, R²={metrics_premium_rf[2]}")
    print(f"Random Forest - TotalClaims: MSE={metrics_claims_rf[0]}, MAE={metrics_claims_rf[1]}, R²={metrics_claims_rf[2]}")
    print(f"XGBoost - TotalPremium: MSE={metrics_premium_xgb[0]}, MAE={metrics_premium_xgb[1]}, R²={metrics_premium_xgb[2]}")
    print(f"XGBoost - TotalClaims: MSE={metrics_claims_xgb[0]}, MAE={metrics_claims_xgb[1]}, R²={metrics_claims_xgb[2]}")

    # Save models
    save_model(model_premium_dt, 'model_premium_dt.pkl')
    save_model(model_claims_dt, 'model_claims_dt.pkl')
    save_model(model_premium_lr, 'model_premium_lr.pkl')
    save_model(model_claims_lr, 'model_claims_lr.pkl')
    save_model(model_premium_rf, 'model_premium_rf.pkl')
    save_model(model_claims_rf, 'model_claims_rf.pkl')
    save_model(model_premium_xgb, 'model_premium_xgb.pkl')
    save_model(model_claims_xgb, 'model_claims_xgb.pkl')

    # Plot evaluation metrics
    plot_metrics(
        [model_premium_dt, model_premium_lr, model_premium_rf, model_premium_xgb], 
        [metrics_premium_dt, metrics_premium_lr, metrics_premium_rf, metrics_premium_xgb], 
        ['Decision Tree', 'Linear Regression', 'Random Forest', 'XGBoost']
    )
    
    # Plot feature importances for tree-based models
    plot_feature_importance(model_premium_dt, X_train_premium)
    plot_feature_importance(model_claims_dt, X_train_claims)
    plot_feature_importance(model_premium_rf, X_train_premium)
    plot_feature_importance(model_claims_rf, X_train_claims)
    plot_feature_importance(model_premium_xgb, X_train_premium)
    plot_feature_importance(model_claims_xgb, X_train_claims)

if __name__ == "__main__":
    main()
