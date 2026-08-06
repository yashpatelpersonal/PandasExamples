import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("testset.csv")

# Display first 10 rows
print(df.head(10))

# Convert datetime column to proper datetime format
df['datetime_utc'] = pd.to_datetime(df['datetime_utc'], errors='coerce')

# Set datetime as index
df.set_index('datetime_utc', inplace=True)

# Choose the value column you want to analyze
# You can change 'hum' to '_tempm', '_pressurem', '_wspdm', etc.
value_column = 'hum'

# Plot raw time series
plt.figure(figsize=(12,6))
plt.plot(df[value_column], label=value_column)
plt.title(f"Humidity Over Time")
plt.xlabel("Date")
plt.ylabel(value_column)
plt.legend()
plt.show()

# Rolling average (30-day)

df['30_day_avg'] = df[value_column].rolling(window=30).mean()

plt.figure(figsize=(12,6))
plt.plot(df[value_column], alpha=0.5, label='Daily Values')
plt.plot(df['30_day_avg'], color='red', label='30-Day Rolling Average')
plt.title(f" Humidity with 30-Day Rolling Average")
plt.xlabel("Date")
plt.ylabel(value_column)
plt.legend()
plt.show()

# Monthly resampling
monthly = df[value_column].resample('ME').mean()

plt.figure(figsize=(12,6))
plt.plot(monthly, marker='o', label='Monthly Average')
plt.title(f"Monthly Average of Humidity ")
plt.xlabel("Month")
plt.ylabel(value_column)
plt.legend()
plt.show()
