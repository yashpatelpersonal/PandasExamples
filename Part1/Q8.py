import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

count_per_company = df['company'].value_counts()
print("Cars per company:\n", count_per_company)
