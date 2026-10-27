import pandas as pd

# Load RFM data
rfm = pd.read_csv("output/rfm_customers.csv")

# Create R score
# Lower recency is better, so recent customers get a higher score
rfm["R_Score"] = pd.qcut(
    rfm["Recency"],
    q=4,
    labels=[4, 3, 2, 1]
)

# Create F score
# Higher frequency is better
rfm["F_Score"] = pd.qcut(
    rfm["Frequency"].rank(method="first"),
    q=4,
    labels=[1, 2, 3, 4]
)

# Create M score
# Higher monetary value is better
rfm["M_Score"] = pd.qcut(
    rfm["Monetary"],
    q=4,
    labels=[1, 2, 3, 4]
)

# Convert scores to integers
rfm["R_Score"] = rfm["R_Score"].astype(int)
rfm["F_Score"] = rfm["F_Score"].astype(int)
rfm["M_Score"] = rfm["M_Score"].astype(int)

# Overall RFM score
rfm["RFM_Score"] = (
    rfm["R_Score"] +
    rfm["F_Score"] +
    rfm["M_Score"]
)

# Create customer segments
def assign_segment(row):
    if row["R_Score"] >= 3 and row["F_Score"] >= 3 and row["M_Score"] >= 3:
        return "Champions"

    elif row["R_Score"] >= 3 and row["F_Score"] >= 2:
        return "Loyal Customers"

    elif row["R_Score"] == 4 and row["F_Score"] <= 2:
        return "Recent Customers"

    elif row["R_Score"] <= 2 and row["F_Score"] >= 3:
        return "At Risk"

    elif row["R_Score"] == 1 and row["F_Score"] <= 2:
        return "Lost Customers"

    else:
        return "Needs Attention"


rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# Save segmented customer data
rfm.to_csv("output/rfm_segmented_customers.csv", index=False)

print("\nSEGMENT COUNTS:")
print(rfm["Segment"].value_counts())

print("\nSAMPLE SEGMENTED CUSTOMERS:")
print(
    rfm[
        [
            "CustomerID",
            "Recency",
            "Frequency",
            "Monetary",
            "R_Score",
            "F_Score",
            "M_Score",
            "Segment"
        ]
    ].head(15)
)