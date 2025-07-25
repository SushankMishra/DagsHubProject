import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml
import pickle
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from mlflow.models import infer_signature
import os
from urllib.parse import urlparse  

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/SushankMishra/DagsHubProject.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "SushankMishra"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "9c8d71ade4339f2f6ae056382c9da10c282d95f8"

params = yaml.safe_load(open("params.yaml"))["train"]


def train_model(input_file, model_output, params):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Split the dataset into features and target variable
    X = df.drop(columns=params["target"])
    y = df[params["target"]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])

    # Initialize the model
    model = RandomForestClassifier(n_estimators=params["n_estimators"], random_state=params["random_state"])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    
    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Save the model
    if not os.path.exists(os.path.dirname(model_output)):
        os.makedirs(os.path.dirname(model_output))
    
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_output}")

def hyperparameter_tuning(input_file, params):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Split the dataset into features and target variable
    X = df.drop(columns=params["target"])
    y = df[params["target"]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=params["random_state"])

    # Define the model
    model = RandomForestClassifier(random_state=params["random_state"])

    param_grid = {
        'n_estimators': params["n_estimators_grid"],
        'max_depth': params["max_depth_grid"],
        'min_samples_split': params["min_samples_split_grid"]
    }

    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_}")