import pandas as pd

# Load full cleaned dataset
df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

print("DATA SHAPE:")
print(df.shape)

print("\nDATE RANGE:")
print(df["InvoiceDate"].min())
print(df["InvoiceDate"].max())

print("\nTOTAL REVENUE:")
print(df["Revenue"].sum())

print("\nUNIQUE CUSTOMERS:")
print(df["CustomerID"].nunique())

print("\nUNIQUE PRODUCTS:")
print(df["StockCode"].nunique())

print("\nTOP 10 COUNTRIES BY REVENUE:")
country_revenue = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(country_revenue)

print("\nTOP 10 PRODUCTS BY QUANTITY:")
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_products)

print("\nTOP 10 CUSTOMERS BY REVENUE:")
top_customers = (
    df.groupby("CustomerID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
print(top_customers)

print("\nORDER COUNT PER CUSTOMER:")
customer_orders = (
    df.groupby("CustomerID")["InvoiceNo"]
    .nunique()
)
print(customer_orders.describe())

print("\nREVENUE DISTRIBUTION:")
print(df["Revenue"].describe())

print("\nQUANTITY DISTRIBUTION:")
print(df["Quantity"].describe())