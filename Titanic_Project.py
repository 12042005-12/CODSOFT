
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump



file = "Titanic-Dataset.csv"

if not os.path.exists(file):
    raise FileNotFoundError(f"{file} not found in folder: {os.getcwd()}")

df = pd.read_csv(file)
print("Dataset loaded. Shape:", df.shape)



df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)


le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])



X = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y = df['Survived']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



model = RandomForestClassifier()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)



dump(model, "titanic_model.joblib")
print("Model saved as titanic_model.joblib")

print("\n--- Completed Successfully ---")
