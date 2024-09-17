import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns

    importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances')
    plt.show()
