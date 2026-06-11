import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

sorted_cars = df.sort_values(by='price')
print("Cars sorted by price:\n", sorted_cars)
