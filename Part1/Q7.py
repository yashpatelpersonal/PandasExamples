import pandas as pd

df = pd.read_csv("Automobile_Cleaned.csv")

toyota_cars = df[df['company'] == 'toyota']
print("Toyota Cars:\n", toyota_cars)
