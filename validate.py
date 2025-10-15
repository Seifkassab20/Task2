import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix

os.makedirs("metrics", exist_ok=True)
model = joblib.load("models/Log.pkl")
df = pd.read_csv("data/test.csv")

x_test = df.drop(columns=['loan_approved', 'name', 'city'])
y_test = df['loan_approved']
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
with open("metrics/metrics.json", "w") as f:
    json.dump({"accuracy DT": acc}, f)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, cmap=plt.cm.magma)
plt.title("Confusion Matrix DT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()
plt.savefig("metrics/confusion_matrix.png")
plt.close()
print("Validation complete.")
