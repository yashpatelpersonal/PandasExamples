import pandas as pd

Car_Price = {
    'Company': ['Toyota', 'Honda', 'BMV', 'Audi'],
    'Price': [23845, 17995, 135925, 71400]
}

Car_Horsepower = {
    'Company': ['Toyota', 'Honda', 'BMV', 'Audi'],
    'horsepower': [141, 80, 182, 160]
}

df_price = pd.DataFrame(Car_Price)
df_hp = pd.DataFrame(Car_Horsepower)

merged_df = pd.merge(df_price, df_hp, on='Company')
print("Merged DataFrame:\n", merged_df)
