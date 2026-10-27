import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load advanced churn dataset
df = pd.read_csv("output/advanced_churn_model_data.csv")

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

# Same 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(
    max_iter=2000,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train_scaled, y_train)

# Get churn probabilities
probabilities = model.predict_proba(X_test_scaled)[:, 1]

# Test different thresholds
thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

results = []

for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    results.append({
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    print("\n==============================")
    print("THRESHOLD:", threshold)
    print("==============================")

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))


results_df = pd.DataFrame(results)

print("\n==============================")
print("THRESHOLD COMPARISON")
print("==============================")

print(results_df)

results_df.to_csv(
    "output/threshold_results.csv",
    index=False
)