import pandas as pd

# Load full cleaned transaction data
df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

cutoff_date = pd.Timestamp("2011-09-01")

# Historical period used to create features
historical = df[df["InvoiceDate"] < cutoff_date].copy()

# Future period used to create churn label
future = df[df["InvoiceDate"] >= cutoff_date].copy()

# --------------------------------------------------
# BASIC CUSTOMER FEATURES
# --------------------------------------------------

customer_features = (
    historical.groupby("CustomerID")
    .agg(
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        TotalQuantity=("Quantity", "sum"),
        UniqueProducts=("StockCode", "nunique")
    )
    .reset_index()
)

customer_features["Recency"] = (
    cutoff_date - customer_features["LastPurchase"]
).dt.days

customer_features["TenureDays"] = (
    cutoff_date - customer_features["FirstPurchase"]
).dt.days

customer_features["AvgOrderValue"] = (
    customer_features["Monetary"] /
    customer_features["Frequency"]
)

customer_features["AvgQuantityPerOrder"] = (
    customer_features["TotalQuantity"] /
    customer_features["Frequency"]
)

# --------------------------------------------------
# RECENT 30, 60, 90 DAY BEHAVIOR
# --------------------------------------------------

last_30_days = historical[
    historical["InvoiceDate"] >= cutoff_date - pd.Timedelta(days=30)
]

last_60_days = historical[
    historical["InvoiceDate"] >= cutoff_date - pd.Timedelta(days=60)
]

last_90_days = historical[
    historical["InvoiceDate"] >= cutoff_date - pd.Timedelta(days=90)
]


def recent_features(data, days):
    return (
        data.groupby("CustomerID")
        .agg(
            **{
                f"Orders_Last_{days}D": ("InvoiceNo", "nunique"),
                f"Revenue_Last_{days}D": ("Revenue", "sum")
            }
        )
        .reset_index()
    )


features_30 = recent_features(last_30_days, 30)
features_60 = recent_features(last_60_days, 60)
features_90 = recent_features(last_90_days, 90)

customer_features = customer_features.merge(
    features_30,
    on="CustomerID",
    how="left"
)

customer_features = customer_features.merge(
    features_60,
    on="CustomerID",
    how="left"
)

customer_features = customer_features.merge(
    features_90,
    on="CustomerID",
    how="left"
)

# Customers with no recent purchases should have zero
recent_columns = [
    "Orders_Last_30D",
    "Revenue_Last_30D",
    "Orders_Last_60D",
    "Revenue_Last_60D",
    "Orders_Last_90D",
    "Revenue_Last_90D"
]

customer_features[recent_columns] = (
    customer_features[recent_columns].fillna(0)
)

# --------------------------------------------------
# ACTIVE MONTHS
# --------------------------------------------------

historical["YearMonth"] = historical["InvoiceDate"].dt.to_period("M")

active_months = (
    historical.groupby("CustomerID")["YearMonth"]
    .nunique()
    .reset_index(name="ActiveMonths")
)

customer_features = customer_features.merge(
    active_months,
    on="CustomerID",
    how="left"
)

# --------------------------------------------------
# AVERAGE DAYS BETWEEN ORDERS
# --------------------------------------------------

orders = (
    historical[
        ["CustomerID", "InvoiceNo", "InvoiceDate"]
    ]
    .drop_duplicates(subset=["CustomerID", "InvoiceNo"])
    .sort_values(["CustomerID", "InvoiceDate"])
)

orders["PreviousOrderDate"] = (
    orders.groupby("CustomerID")["InvoiceDate"].shift(1)
)

orders["DaysBetweenOrders"] = (
    orders["InvoiceDate"] - orders["PreviousOrderDate"]
).dt.days

avg_days_between = (
    orders.groupby("CustomerID")["DaysBetweenOrders"]
    .mean()
    .reset_index(name="AvgDaysBetweenOrders")
)

customer_features = customer_features.merge(
    avg_days_between,
    on="CustomerID",
    how="left"
)

# Customers with only one order have no gap,
# so fill with their tenure
customer_features["AvgDaysBetweenOrders"] = (
    customer_features["AvgDaysBetweenOrders"]
    .fillna(customer_features["TenureDays"])
)

# --------------------------------------------------
# CREATE CHURN TARGET
# --------------------------------------------------

future_customers = set(future["CustomerID"].unique())

customer_features["Churn"] = (
    ~customer_features["CustomerID"].isin(future_customers)
).astype(int)

# Remove date columns
customer_features = customer_features.drop(
    columns=["FirstPurchase", "LastPurchase"]
)

# Save dataset
customer_features.to_csv(
    "output/advanced_churn_model_data.csv",
    index=False
)

print("ADVANCED DATASET SHAPE:")
print(customer_features.shape)

print("\nFEATURES:")
print(customer_features.columns.tolist())

print("\nCHURN DISTRIBUTION:")
print(customer_features["Churn"].value_counts())

print("\nFIRST 10 ROWS:")
print(customer_features.head(10))

print("\nMISSING VALUES:")
print(customer_features.isnull().sum())