import matplotlib.pyplot as plt
import seaborn as sns

def plot_postalcode_premium(df):
    premium_by_postalcode = df.groupby('PostalCode')['TotalPremium'].mean().sort_values()
    plt.figure(figsize=(10, 15))
    premium_by_postalcode.plot(kind='barh', color='skyblue')
    plt.title('Average Total Premium by Postal Code')
    plt.xlabel('Average Premium')
    plt.ylabel('PostalCode')
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_premium_vs_claims(df):
    plt.figure(figsize=(10, 6))
    sns.regplot(x='TotalPremium', y='TotalClaims', data=df, scatter_kws={'alpha':0.3})
    plt.title('Correlation Between Premium and Claims')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.show()
