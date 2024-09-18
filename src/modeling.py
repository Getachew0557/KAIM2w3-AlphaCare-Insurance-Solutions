import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Function to handle train-test split for premium and claims
def split_data(X, y_premium, y_claims, test_size=0.2, random_state=42):
    X_train_premium, X_test_premium, y_train_premium, y_test_premium = train_test_split(X, y_premium, test_size=test_size, random_state=random_state)
    X_train_claims, X_test_claims, y_train_claims, y_test_claims = train_test_split(X, y_claims, test_size=test_size, random_state=random_state)
    return (X_train_premium, X_test_premium, y_train_premium, y_test_premium, 
            X_train_claims, X_test_claims, y_train_claims, y_test_claims)

def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    return model
