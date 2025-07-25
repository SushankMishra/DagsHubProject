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

