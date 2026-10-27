import pandas as pd

# Load raw dataset
df = pd.read_excel("data/online+retail.csv/Online Retail.xlsx")

print("Original shape:", df.shape)

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove rows with missing Description
df = df.dropna(subset=["Description"])

# Remove cancelled transactions
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only positive quantities
df = df[df["Quantity"] > 0]

# Keep only positive unit prices
df = df[df["UnitPrice"] > 0]

# Convert CustomerID to integer
df["CustomerID"] = df["CustomerID"].astype(int)

# Create revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

df.to_csv("data/cleaned_full_ecommerce_data.csv", index=False)

# Take 60,000-row sample
sample_df = df.sample(n=60000, random_state=42)

# Save cleaned sample
sample_df.to_csv("data/cleaned_ecommerce_data.csv", index=False)

print("Cleaned full dataset shape:", df.shape)
print("Sample shape:", sample_df.shape)

print("\nMissing values after cleaning:")
print(sample_df.isnull().sum())

print("\nSample preview:")
print(sample_df.head())

print("\nUNIQUE CUSTOMERS IN SAMPLE:")
print(sample_df["CustomerID"].nunique())

print("\nDATE RANGE IN SAMPLE:")
print(sample_df["InvoiceDate"].min())
print(sample_df["InvoiceDate"].max())

print("\nTOP 10 COUNTRIES:")
print(sample_df["Country"].value_counts().head(10))

print("\nQUANTITY STATISTICS:")
print(sample_df["Quantity"].describe())

print("\nREVENUE STATISTICS:")
print(sample_df["Revenue"].describe())

print("\nCUSTOMER PURCHASE COUNTS:")
print(sample_df.groupby("CustomerID")["InvoiceNo"].nunique().describe())