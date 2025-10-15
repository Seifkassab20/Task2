import os
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
os.makedirs('models', exist_ok=True)
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['loan_approved', 'name', 'city'])
y = df['loan_approved']

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, "models/Log.pkl")
print("Tarining complete")



