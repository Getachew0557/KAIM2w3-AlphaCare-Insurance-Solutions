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

