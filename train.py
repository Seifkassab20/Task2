import os
import yaml
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

model_name = params["model"]["name"]

df = pd.read_csv("data/train.csv")
X = df.drop(columns=["loan_approved" , "name", "city"])
y = df["loan_approved"]

if model_name == "logistic_regression":
    model = LogisticRegression(max_iter=1000, random_state=42)
else :
    model = DecisionTreeClassifier(random_state=42)

model.fit(X, y)

os.makedirs("models", exist_ok=True)
model_path = f"models/{model_name}.pkl"
joblib.dump(model, model_path)

