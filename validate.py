import os
import pandas as pd
import joblib
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

os.makedirs("models", exist_ok=True)
df_train = pd.read_csv("data/train.csv")
df_val = pd.read_csv("data/test.csv")
x_train = df_train.drop(columns=['loan_approved', 'name', 'city'])
y_train = df_train['loan_approved']
x_val = df_val.drop(columns=['loan_approved', 'name', 'city'])
y_val = df_val['loan_approved']

log = LogisticRegression(random_state=42)
log.fit(x_train, y_train)
y_pred_log = log.predict(x_val) 
acc_log = accuracy_score(y_val, y_pred_log)

joblib.dump(log, "models/logistic_regression_model.pkl")
metrics = {"logistic_regression_accuracy": acc_log}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Validation complete.")
