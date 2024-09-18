from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2

def plot_metrics(models, metrics, labels):
    """ Plot MSE, MAE, and R² for multiple models """
    mse_vals, mae_vals, r2_vals = zip(*metrics)
    
    plt.figure(figsize=(18, 5))
    
    # MSE Plot
    plt.subplot(1, 3, 1)
    plt.bar(labels, mse_vals, color='blue')
    plt.title('MSE')
    plt.ylabel('Mean Squared Error')

    # MAE Plot
    plt.subplot(1, 3, 2)
    plt.bar(labels, mae_vals, color='orange')
    plt.title('MAE')
    plt.ylabel('Mean Absolute Error')

    # R² Plot
    plt.subplot(1, 3, 3)
    plt.bar(labels, r2_vals, color='green')
    plt.title('R² Score')
    plt.ylabel('R² Score')

    plt.tight_layout()
    plt.show()
