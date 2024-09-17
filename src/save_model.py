import os
import joblib

def save_model(model, model_name, model_dir='../notebook/model/'):
    """
    Save the trained model to a file.

    Parameters:
    - model: The trained model object to save.
    - model_name: The name of the file to save the model as.
    - model_dir: The directory where the model will be saved.
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
