import pandas as pd

# Load the dataset
df = pd.read_excel("data/online+retail.csv/Online Retail.xlsx")

# Show first 5 rows
print("\nFIRST 5 ROWS:")
print(df.head())

# Show number of rows and columns
print("\nSHAPE:")
print(df.shape)

# Show column names
print("\nCOLUMNS:")
print(df.columns.tolist())

# Show data types and non-null counts
print("\nDATA INFO:")
df.info()

# Show missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Show duplicate rows count
print("\nDUPLICATE ROWS:")
print(df.duplicated().sum())

# Show basic statistics
print("\nBASIC STATISTICS:")
print(df.describe())