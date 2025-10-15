import os
import pandas as pd
import joblib
import json
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['loan_approved', 'name', 'city'])
y = df['loan_approved']

model = LogisticRegression(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy}, f)
print("Tarining complete")



