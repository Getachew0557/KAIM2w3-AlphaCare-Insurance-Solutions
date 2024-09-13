import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def check_missing_values(df):
    print(df.isnull().sum())
    return df.isnull().sum()

def describe_data(df):
    return df.describe()

def plot_missing_values(df):
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) == 0:
        print("No missing values.")
        return
    
    missing_values_df = pd.DataFrame({
        'Column': columns_with_missing.index,
        'Missing Values': columns_with_missing.values
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    table = ax.table(
        cellText=missing_values_df.values,
        colLabels=['Column', 'Missing Values'],
        cellLoc='center',
        loc='center',
        cellColours=[['lightgrey'] * 2] * len(missing_values_df),
        colColours=['#d9d9d9'] * 2
    )
    plt.title('Columns with Missing Values', fontsize=14)
    plt.show()

def plot_histograms(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plt.figure(figsize=(16, 12))
    for i, col in enumerate(numerical_cols):
        plt.subplot(4, 4, i + 1)
        sns.histplot(df[col].dropna(), bins=20, kde=False)
        plt.title(col)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

def plot_categorical_bars(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].value_counts().plot(kind='bar', figsize=(10, 5))
        plt.title(f'Distribution of {col}')
        plt.show()

def plot_correlation_matrix(df):
    correlation_matrix = df[['TotalPremium', 'TotalClaims', 'PostalCode']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    
def data_compression(df):
    """Compress the DataFrame to reduce memory usage."""
    initial_memory = df.memory_usage().sum() / (1024 ** 2)
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'object':
            df[col] = df[col].astype('category')
        elif col_type in ['int64', 'float64']:
            df[col] = pd.to_numeric(df[col], downcast='integer' if col_type == 'int64' else 'float')
    
    final_memory = df.memory_usage().sum() / (1024 ** 2)
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Reduced memory usage by: {initial_memory - final_memory:.2f} MB")
