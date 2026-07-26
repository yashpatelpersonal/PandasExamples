import pandas as pd
import seaborn as sns

# Load dataset
df = sns.load_dataset("iris")

# Display first 10 rows
print("First 10 rows of the dataset:")
print(df.head(10))

# Aggregation functions
print("Count:\n", df.count())
print("\nFirst:\n", df.iloc[0])
print("\nLast:\n", df.iloc[-1])
print("\nMean:\n", df.mean(numeric_only=True))
print("\nMedian:\n", df.median(numeric_only=True))
print("\nMin:\n", df.min(numeric_only=True))
print("\nMax:\n", df.max(numeric_only=True))
print("\nStandard Deviation:\n", df.std(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))
print("\nSum:\n", df.sum(numeric_only=True))

# Groupby analysis
grouped = df.groupby("species").mean()
print("\nGroupby (species) Mean Values:")
print(grouped)
