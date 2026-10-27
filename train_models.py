import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load churn modeling dataset
df = pd.read_csv("output/churn_model_data.csv")

# CustomerID is an identifier, not a predictive feature
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

print("FEATURES USED:")
print(X.columns.tolist())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING ROWS:", X_train.shape[0])
print("TESTING ROWS:", X_test.shape[0])

# --------------------------------------------------
# LOGISTIC REGRESSION
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, logistic_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        logistic_predictions
    )
)

# --------------------------------------------------
# RANDOM FOREST
# --------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, rf_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_predictions
    )
)