import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

most_expensive_each = df.loc[df.groupby('company')['price'].idxmax()]
print("Most expensive car per company:\n", most_expensive_each[['company', 'price']])
