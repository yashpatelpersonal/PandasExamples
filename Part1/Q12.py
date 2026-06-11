import pandas as pd

GermanCars = {
    'Company': ['Ford', 'Mercedes', 'BMV', 'Audi'],
    'Price': [23845, 171995, 135925, 71400]
}

japaneseCars = {
    'Company': ['Toyota', 'Honda', 'Nissan', 'Mitsubishi'],
    'Price': [29995, 23600, 61500, 58900]
}

df_german = pd.DataFrame(GermanCars)
df_japanese = pd.DataFrame(japaneseCars)

df_combined = pd.concat([df_german, df_japanese], ignore_index=True)
print("Combined DataFrame:\n", df_combined)
