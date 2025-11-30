
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv")

print("Dataset Loaded Successfully!")
print(df.head())
print(df.info())
print("\nClass Distribution:")
print(df["Class"].value_counts())
X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining and Testing Split Completed!")
print("\nBefore SMOTE:", y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

print("After SMOTE:", y_train_res.value_counts())
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train_res)
X_test = scaler.transform(X_test)

print("\nScaling Completed!")
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_res, y_train_res)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_res, y_train_res)

print("\nModel Training Completed!")
y_pred_log = log_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
print("\n=============================")
print("LOGISTIC REGRESSION REPORT")
print("=============================")
print(classification_report(y_test, y_pred_log))

print("\n=============================")
print("RANDOM FOREST REPORT")
print("=============================")
print(classification_report(y_test, y_pred_rf))
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\nPROJECT COMPLETED SUCCESSFULLY!")
