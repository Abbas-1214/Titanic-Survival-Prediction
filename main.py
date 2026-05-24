import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("train.csv")

# Select useful columns
df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]

# Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Fare"] = df["Fare"].fillna(df["Fare"].mean())

# Convert male/female to numbers
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# Features and target
X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Example prediction
prediction = model.predict([[1, 1, 25, 50]])

if prediction[0] == 1:
    print("Prediction: Survived")
else:
    print("Prediction: Did Not Survive")

# Accuracy graph
plt.bar(["Decision Tree"], [accuracy * 100])

plt.title("Titanic Survival Prediction Accuracy")
plt.ylabel("Accuracy (%)")

plt.show()
print(df.head())
print(df.info())