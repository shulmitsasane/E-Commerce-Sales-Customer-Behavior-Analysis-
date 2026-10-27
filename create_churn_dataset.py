import pandas as pd

# Load full cleaned transaction data
df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

# --------------------------------------------------
# 1. Define observation and prediction periods
# --------------------------------------------------

cutoff_date = pd.Timestamp("2011-09-01")

# Data BEFORE September is used to create customer features
historical = df[df["InvoiceDate"] < cutoff_date].copy()

# Data FROM September onward is used to check
# whether the customer purchased again
future = df[df["InvoiceDate"] >= cutoff_date].copy()

print("Historical transactions:", historical.shape)
print("Future transactions:", future.shape)

# --------------------------------------------------
# 2. Create customer features from historical data
# --------------------------------------------------

customer_features = (
    historical.groupby("CustomerID")
    .agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique")
    )
    .reset_index()
)

# Recency = days since last purchase before cutoff
customer_features["Recency"] = (
    cutoff_date - customer_features["LastPurchase"]
).dt.days

# Tenure = how long the customer has been active
customer_features["TenureDays"] = (
    cutoff_date - customer_features["FirstPurchase"]
).dt.days

# Average order value
customer_features["AvgOrderValue"] = (
    customer_features["Monetary"] /
    customer_features["Frequency"]
)

# Average quantity per order
customer_features["AvgQuantityPerOrder"] = (
    customer_features["TotalQuantity"] /
    customer_features["Frequency"]
)

# --------------------------------------------------
# 3. Determine which customers purchased in future
# --------------------------------------------------

future_customers = set(future["CustomerID"].unique())

# Churn:
# 0 = customer purchased again
# 1 = customer did NOT purchase again
customer_features["Churn"] = (
    ~customer_features["CustomerID"].isin(future_customers)
).astype(int)

# --------------------------------------------------
# 4. Remove date columns that models won't use
# --------------------------------------------------

customer_features = customer_features.drop(
    columns=["LastPurchase", "FirstPurchase"]
)

# --------------------------------------------------
# 5. Save modeling dataset
# --------------------------------------------------

customer_features.to_csv(
    "output/churn_model_data.csv",
    index=False
)

print("\nCUSTOMER DATASET SHAPE:")
print(customer_features.shape)

print("\nFIRST 10 CUSTOMERS:")
print(customer_features.head(10))

print("\nCHURN DISTRIBUTION:")
print(customer_features["Churn"].value_counts())

print("\nCHURN PERCENTAGE:")
print(
    customer_features["Churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFEATURE STATISTICS:")
print(customer_features.describe())