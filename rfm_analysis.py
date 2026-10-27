import pandas as pd

# Load the full cleaned transaction dataset
df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

# Use one day after the latest transaction as the analysis date
snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

print("Snapshot date:", snapshot_date)

# Create one row per customer
rfm = (
    df.groupby("CustomerID")
    .agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    )
    .reset_index()
)

# Save customer-level RFM data
rfm.to_csv("output/rfm_customers.csv", index=False)

print("\nRFM SHAPE:")
print(rfm.shape)

print("\nFIRST 10 CUSTOMERS:")
print(rfm.head(10))

print("\nRFM STATISTICS:")
print(rfm[["Recency", "Frequency", "Monetary"]].describe())