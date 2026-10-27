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

# Load advanced churn dataset
df = pd.read_csv("output/advanced_churn_model_data.csv")

# CustomerID is only an identifier
X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

print("FEATURES USED:")
print(X.columns.tolist())

# Use same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAINING ROWS:", X_train.shape[0])
print("TESTING ROWS:", X_test.shape[0])

# ==================================================
# LOGISTIC REGRESSION
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

logistic_model.fit(X_train_scaled, y_train)

logistic_predictions = logistic_model.predict(X_test_scaled)

print("\n================================")
print("ADVANCED LOGISTIC REGRESSION")
print("================================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, logistic_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions))


# ==================================================
# RANDOM FOREST
# ==================================================

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

print("\n================================")
print("ADVANCED RANDOM FOREST")
print("================================")

print(
    "Accuracy:",
    round(accuracy_score(y_test, rf_predictions), 4)
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))

print("\nClassification Report:")
print(classification_report(y_test, rf_predictions))


# ==================================================
# FEATURE IMPORTANCE
# ==================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\n================================")
print("TOP RANDOM FOREST FEATURES")
print("================================")

print(feature_importance.head(10))

feature_importance.to_csv(
    "output/random_forest_feature_importance.csv",
    index=False
)