import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

avg_mileage = df.groupby('company')['average-mileage'].mean()
print("Average mileage per company:\n", avg_mileage)
