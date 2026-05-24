import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("train.csv")

print("Dataset Loaded Successfully!")

print("\nFirst Five Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values Analysis")
print(df.isnull().sum())

df = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "Fare"
    ]
]

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

df["Sex"] = df["Sex"].map(
    {
        "male": 0,
        "female": 1
    }
)

print("\nSurvival Distribution")
print(df["Survived"].value_counts())

plt.figure(figsize=(6, 4))

df["Survived"].value_counts().plot(
    kind="bar"
)

plt.title("Survival Distribution")
plt.xlabel("0 = Died, 1 = Survived")
plt.ylabel("Passengers")

plt.show()

X = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "Fare"
    ]
]

y = df["Survived"]

print("\nFeatures:")
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records :", len(X_train))
print("Testing Records  :", len(X_test))

model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed")

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "\nAccuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(6, 4))

plt.bar(
    ["Decision Tree"],
    [accuracy * 100]
)

plt.title("Model Accuracy")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.show()

plt.figure(figsize=(5, 4))

plt.imshow(cm)

plt.title("Confusion Matrix")
plt.colorbar()

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

importance = model.feature_importances_

feature_names = [
    "Pclass",
    "Sex",
    "Age",
    "Fare"
]

print("\nFeature Importance")

for feature, score in zip(
    feature_names,
    importance
):
    print(
        feature,
        ":",
        round(score, 4)
    )

plt.figure(figsize=(7, 4))

plt.bar(
    feature_names,
    importance
)

plt.title("Feature Importance")
plt.ylabel("Importance Score")

plt.show()

sample_passenger = pd.DataFrame(
    [[1, 1, 25, 50]],
    columns=[
        "Pclass",
        "Sex",
        "Age",
        "Fare"
    ]
)

prediction = model.predict(
    sample_passenger
)

print("\nExample Passenger")
print("Class: 1")
print("Gender: Female")
print("Age: 25")
print("Fare: 50")

if prediction[0] == 1:
    print("Prediction: SURVIVED")
else:
    print("Prediction: DID NOT SURVIVE")

try:

    pclass = int(
        input(
            "\nEnter Passenger Class (1-3): "
        )
    )

    gender = input(
        "Enter Gender (male/female): "
    ).lower()

    age = float(
        input(
            "Enter Age: "
        )
    )

    fare = float(
        input(
            "Enter Fare: "
        )
    )

    if gender == "male":
        gender_value = 0
    else:
        gender_value = 1

    result = model.predict(
        [
            [
                pclass,
                gender_value,
                age,
                fare
            ]
        ]
    )

    print("\nPrediction Result")

    if result[0] == 1:
        print("Passenger is likely to SURVIVE")
    else:
        print("Passenger is likely NOT to survive")

except:
    print("Invalid Input Entered")

print("\nProject Execution Completed Successfully")