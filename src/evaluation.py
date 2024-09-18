import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_metrics(metrics_dict, model_name):
    plt.figure(figsize=(15, 5))
    
    for i, (metric, values) in enumerate(metrics_dict.items()):
        plt.subplot(1, len(metrics_dict), i + 1)
        plt.bar(model_name, values, color=['blue', 'orange', 'green'][i])
        plt.title(f'{metric} for {model_name}')
        plt.ylabel(metric)
    
    plt.tight_layout()
    plt.show()

def evaluate_and_plot(model, X_test, y_test, model_name):
    mse, mae, r2 = evaluate_model(model, X_test, y_test)
    print(f"{model_name} - MSE: {mse}, MAE: {mae}, R^2: {r2}")

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R^2': r2
    }
    plot_metrics(metrics, model_name)
