import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import yaml
import mlflow
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



def hyperparameter_tuning(x_train,y_train, param_grid):
    model = RandomForestClassifier(random_state=params["random_state"])
    # Perform grid search
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)
    return grid_search

def train_model(data_path,model_path, random_state,n_estimators, max_depth):
    # Load the dataset
    df = pd.read_csv(data_path)
    
    # Split the data into features and target
    X = df.drop(columns=['Purchased'])
    y = df['Purchased']
    
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=params["test_size"], random_state=random_state)
    
    with mlflow.start_run():
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [n_estimators],
            'max_depth': [max_depth]
        }
        grid_search = hyperparameter_tuning(x_train, y_train, param_grid)
        
        # Get the best model
        best_model = grid_search.best_estimator_
        mlflow.log_params({
            "best_n_estimators": grid_search.best_params_['n_estimators'],
            "best_max_depth": grid_search.best_params_['max_depth'],
            "random_state": random_state
        })  
        # Train the model
        best_model.fit(x_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(x_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", accuracy)
        # Save the model
        
        # Log the model with MLflow
        signature = infer_signature(x_train, y_pred)
        # mlflow.log_artifact(model_path)
        # Log the classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_text(str(report), "classification_report.txt")
        # Log the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_text(str(cm),"confusion_matrix.txt")

        tracking_url_type_store = urlparse(os.environ["MLFLOW_TRACKING_URI"]).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(best_model, "model", signature=signature)
        else:
            mlflow.pyfunc.log_model("model", python_model=best_model, signature=signature)
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

if __name__ == "__main__":
    data_path = params["input"]
    model_path = params["output"]
    random_state = params["random_state"]
    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    
    if not os.path.exists(data_path):
        print(f"Data file {data_path} does not exist.")
        exit(1)
    
    train_model(data_path, model_path, random_state, n_estimators, max_depth)