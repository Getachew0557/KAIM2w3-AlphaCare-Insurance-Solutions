import pandas as pd
import matplotlib.pyplot as plt

def handle_missing_values(df):
    # Drop columns with very high missing values
    cols_to_drop = ['NumberOfVehiclesInFleet', 'CrossBorder', 'CustomValueEstimate', 
                    'Converted', 'Rebuilt', 'WrittenOff']
    df = df.drop(columns=cols_to_drop)

    # Impute categorical columns
    categorical_cols = ['NewVehicle', 'AccountType', 'Bank', 'VehicleType', 
                        'make', 'Model', 'mmcode', 'bodytype']
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Impute numerical columns
    numerical_cols = ['cubiccapacity', 'kilowatts', 'Cylinders', 'NumberOfDoors']
    for col in numerical_cols:
        df[col].fillna(df[col].mean(), inplace=True)

    # Drop rows with missing values in a specific column
    df.dropna(subset=['CapitalOutstanding'], inplace=True)
    
    # Drop additional columns
    columns_to_drop = ['NewVehicle', 'VehicleType', 'make', 'Model', 
                       'bodytype', 'Cylinders', 'cubiccapacity', 'RegistrationYear', 
                       'VehicleIntroDate', 'kilowatts', 'NumberOfDoors']
    df = df.drop(columns=columns_to_drop)

    # Remove duplicates
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df

def get_numerical_columns(df):
    return df.select_dtypes(include=['float64', 'int64']).columns.tolist()

def convert_datetime_features(df):
    # Convert 'TransactionMonth' and 'VehicleIntroDate' to datetime
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df['VehicleIntroDate'] = pd.to_datetime(df['VehicleIntroDate'], errors='coerce')

    # Extract features from datetime columns
    df['TransactionMonth_Year'] = df['TransactionMonth'].dt.year
    df['TransactionMonth_Month'] = df['TransactionMonth'].dt.month
    df['VehicleIntroDate_Year'] = df['VehicleIntroDate'].dt.year
    df['VehicleIntroDate_Month'] = df['VehicleIntroDate'].dt.month

    # Drop original datetime columns
    df = df.drop(columns=['TransactionMonth', 'VehicleIntroDate'])

    return df

def encode_categorical_features(df):
    # One-hot encode categorical variables
    categorical_cols = ['NewVehicle', 'AccountType', 'Bank', 'VehicleType', 
                        'make', 'Model', 'mmcode', 'bodytype']
    
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


