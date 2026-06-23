# -------------------------------------------
# Lab 6 - Chipotle Analysis
# -------------------------------------------

import pandas as pd

# 1. Load the dataset
df = pd.read_excel("Lab 6 - Chipotle.xls")

# 2. Convert item_price to numeric
# Remove the dollar sign and convert to float
df["item_price"] = df["item_price"].replace('[\\$,]', '', regex=True).astype(float)

# 3. Average price of items containing "chicken"
chicken_avg = df[df["item_name"].str.contains("chicken", case=False)]["item_price"].mean()
print("Average Chicken Item Price:", chicken_avg)

# 4. Average price of items containing "steak"
steak_avg = df[df["item_name"].str.contains("steak", case=False)]["item_price"].mean()
print("Average Steak Item Price:", steak_avg)

# 5. Which produced more revenue?
chicken_revenue = df[df["item_name"].str.contains("chicken", case=False)]["item_price"].sum()
steak_revenue = df[df["item_name"].str.contains("steak", case=False)]["item_price"].sum()

print("Chicken Revenue:", chicken_revenue)
print("Steak Revenue:", steak_revenue)

if chicken_revenue > steak_revenue:
    print("Chicken items produced more revenue.")
else:
    print("Steak items produced more revenue.")

# 6. Missing values
total_missing = df.isna().sum().sum()
missing_by_column = df.isna().sum()

print("\nTotal Missing Values:", total_missing)
print("\nMissing Values by Column:")
print(missing_by_column)
