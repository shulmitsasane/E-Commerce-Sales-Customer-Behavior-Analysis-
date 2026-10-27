import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load advanced churn dataset
df = pd.read_csv("output/advanced_churn_model_data.csv")

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Same 80/20 split for fair comparison
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Gradient Boosting model
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Train model
gb_model.fit(X_train, y_train)

# Predict
predictions = gb_model.predict(X_test)

print("\n==============================")
print("GRADIENT BOOSTING")
print("==============================")

print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb_model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP GRADIENT BOOSTING FEATURES:")
print(feature_importance.head(10))

feature_importance.to_csv(
    "output/gradient_boosting_feature_importance.csv",
    index=False
)