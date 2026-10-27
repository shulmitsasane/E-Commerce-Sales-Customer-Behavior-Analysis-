import pandas as pd

df = pd.read_csv(
    "data/cleaned_full_ecommerce_data.csv",
    parse_dates=["InvoiceDate"]
)

print("\nTOP 10 ROWS BY QUANTITY:")
print(
    df.nlargest(10, "Quantity")[
        [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "CustomerID",
            "Revenue"
        ]
    ]
)

print("\nTOP 10 ROWS BY REVENUE:")
print(
    df.nlargest(10, "Revenue")[
        [
            "InvoiceNo",
            "StockCode",
            "Description",
            "Quantity",
            "UnitPrice",
            "CustomerID",
            "Revenue"
        ]
    ]
)