import pandas as pd

df = pd.read_excel("Lab 4 - Automobile.xls")

df_clean = df.replace(['?', 'n.a'], pd.NA).dropna()
df_clean.to_csv("Automobile_Cleaned.csv", index=False)

print("Cleaned dataset saved as Automobile_Cleaned.csv")
