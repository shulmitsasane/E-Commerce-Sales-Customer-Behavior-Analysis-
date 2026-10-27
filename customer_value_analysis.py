import pandas as pd

# --------------------------------------------------
# LOAD CLEANED DATA
# --------------------------------------------------

df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

# --------------------------------------------------
# MONTHLY REVENUE
# --------------------------------------------------

df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_revenue = (
    df.groupby("YearMonth")
    .agg(
        Revenue=("Revenue", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique")
    )
    .reset_index()
)

monthly_revenue.to_csv(
    "output/monthly_revenue.csv",
    index=False
)

# --------------------------------------------------
# CUSTOMER VALUE / CLV PROXY
# --------------------------------------------------

snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

customer_value = (
    df.groupby("CustomerID")
    .agg(
        TotalRevenue=("Revenue", "sum"),
        TotalOrders=("InvoiceNo", "nunique"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max")
    )
    .reset_index()
)

# Average amount spent per order
customer_value["AvgOrderValue"] = (
    customer_value["TotalRevenue"] /
    customer_value["TotalOrders"]
)

# Number of months the customer was observed in the dataset
customer_value["ObservedMonths"] = (
    (snapshot_date - customer_value["FirstPurchase"]).dt.days / 30
).clip(lower=1)

# Average number of orders per month
customer_value["MonthlyPurchaseFrequency"] = (
    customer_value["TotalOrders"] /
    customer_value["ObservedMonths"]
)

# Annualized CLV proxy
customer_value["CLV"] = (
    customer_value["AvgOrderValue"] *
    customer_value["MonthlyPurchaseFrequency"] *
    12
)

customer_value.to_csv(
    "output/customer_clv.csv",
    index=False
)

# --------------------------------------------------
# PRODUCT PERFORMANCE
# --------------------------------------------------

product_performance = (
    df.groupby(["StockCode", "Description"])
    .agg(
        Revenue=("Revenue", "sum"),
        QuantitySold=("Quantity", "sum"),
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique")
    )
    .reset_index()
    .sort_values("Revenue", ascending=False)
)

product_performance.to_csv(
    "output/product_performance.csv",
    index=False
)

# --------------------------------------------------
# FILTER REAL PRODUCTS FOR DASHBOARD
# --------------------------------------------------

excluded_descriptions = [
    "POSTAGE",
    "Manual"
]

dashboard_products = product_performance[
    ~product_performance["Description"].isin(excluded_descriptions)
].copy()

dashboard_products.to_csv(
    "output/dashboard_product_performance.csv",
    index=False
)

# --------------------------------------------------
# SUMMARY METRICS
# --------------------------------------------------

total_revenue = df["Revenue"].sum()

order_values = (
    df.groupby("InvoiceNo")["Revenue"]
    .sum()
)

average_order_value = order_values.mean()

unique_customers = df["CustomerID"].nunique()

unique_orders = df["InvoiceNo"].nunique()

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\nTOTAL REVENUE:")
print(round(total_revenue, 2))

print("\nAVERAGE ORDER VALUE:")
print(round(average_order_value, 2))

print("\nUNIQUE CUSTOMERS:")
print(unique_customers)

print("\nTOTAL ORDERS:")
print(unique_orders)

print("\nTOP 10 CUSTOMERS BY CLV:")
print(
    customer_value[
        [
            "CustomerID",
            "TotalRevenue",
            "TotalOrders",
            "AvgOrderValue",
            "MonthlyPurchaseFrequency",
            "CLV"
        ]
    ]
    .sort_values("CLV", ascending=False)
    .head(10)
)

print("\nMONTHLY REVENUE:")
print(monthly_revenue)

print("\nTOP 10 REAL PRODUCTS BY REVENUE:")
print(
    dashboard_products[
        [
            "StockCode",
            "Description",
            "Revenue",
            "QuantitySold",
            "Orders",
            "Customers"
        ]
    ]
    .head(10)
)