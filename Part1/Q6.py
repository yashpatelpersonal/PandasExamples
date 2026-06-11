import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

max_price = df['price'].max()
most_expensive = df[df['price'] == max_price][['company', 'price']]

print("Most expensive car:\n", most_expensive)
