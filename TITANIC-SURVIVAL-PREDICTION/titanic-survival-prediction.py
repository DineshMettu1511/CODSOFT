#Titanic Survival Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)



df = pd.read_csv("titanic.csv")


print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())


df["Age"] = df["Age"].fillna(df["Age"].mean())


df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

df.drop("Cabin", axis=1, inplace=True)



label_encoder = LabelEncoder()


df["Sex"] = label_encoder.fit_transform(df["Sex"])


df["Embarked"] = label_encoder.fit_transform(
    df["Embarked"]
)


df.drop(
    ["PassengerId", "Name", "Ticket"],
    axis=1,
    inplace=True
)



X = df.drop("Survived", axis=1)


y = df["Survived"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression(max_iter=1000)


model.fit(X_train, y_train)



y_pred = model.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========\n")
print("Accuracy :", accuracy * 100, "%")



print("\n========== CLASSIFICATION REPORT ==========\n")
print(classification_report(y_test, y_pred))



cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========\n")
print(cm)



plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()



# Survival count
plt.figure(figsize=(6, 4))

sns.countplot(x="Survived", data=df)

plt.title("Survival Count")

plt.show()



plt.figure(figsize=(6, 4))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title("Survival Based on Gender")

plt.show()



sample_data = np.array([
    [1, 0, 25, 0, 0, 100, 0]
])

prediction = model.predict(sample_data)

print("\n========== CUSTOM PREDICTION ==========\n")

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger Did Not Survive")

