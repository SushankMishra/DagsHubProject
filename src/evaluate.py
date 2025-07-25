import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml
import mlflow
import pickle
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models import infer_signature
import os, sys
from urllib.parse import urlparse  

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/SushankMishra/DagsHubProject.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "SushankMishra"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9c8d71ade4339f2f6ae056382c9da10c282d95f8"

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Split the data into features and target
    X = df.drop(columns=['Purchased'])
    y = df['Purchased']
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    # Load the trained model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Make predictions
    y_pred = model.predict(X)
    
    # Evaluate the model
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    mlflow.log_metrics({
        "accuracy": accuracy
    })

if __name__ == "__main__":
    data_path = params["input"]
    model_path = params["output"]
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} does not exist.")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Model file {model_path} does not exist.")
        sys.exit(1)
    
    evaluate(data_path, model_path)