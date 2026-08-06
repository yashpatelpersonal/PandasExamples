# ---------------------------------------------------------
# Lab 8 – Treatment Data Analysis
# Author: Yash
# ---------------------------------------------------------

import pandas as pd

# ---------------------------------------------------------
# 1. READ DATA
# ---------------------------------------------------------

df = pd.read_excel("Lab 8 - Treatment.xls")
print("\n--- Dataset Loaded ---")
print(df.head())

# ---------------------------------------------------------
# 2. GROUP BY TREATMENT
# ---------------------------------------------------------

treat_group = df.groupby("Treatment")
print("\n--- Grouped by Treatment ---")
print(treat_group.size())

# ---------------------------------------------------------
# 3. DESCRIBE RELATIVE FITNESS BY TREATMENT
# ---------------------------------------------------------

print("\n--- Relative Fitness Distribution by Treatment ---")
desc_treat = treat_group["RelativeFitness"].describe()
print(desc_treat)

# Summary (human-written):
# You will see differences in spread and central values.
# Some treatments may show higher median fitness or wider ranges.
# Look for skew, outliers, or treatments with consistently higher values.

# ---------------------------------------------------------
# 4. GROUP BY TREATMENT AND GROUP
# ---------------------------------------------------------

treat_group2 = df.groupby(["Treatment", "Group"])
desc_treat_group = treat_group2["RelativeFitness"].describe()

print("\n--- Relative Fitness by Treatment + Group ---")
print(desc_treat_group)

# Summary:
# Compare groups inside each treatment.
# If patterns repeat (e.g., Group A always higher), differences are consistent.
# If values shift across treatments, groups respond differently.

# ---------------------------------------------------------
# 5. AGGREGATION FUNCTIONS
# ---------------------------------------------------------

agg_results = treat_group["RelativeFitness"].agg(
    ["mean", "max", "median", "sum", "std"]
)

print("\n--- Aggregation Results ---")
print(agg_results)

# ---------------------------------------------------------
# 6. ADDITIONAL OBSERVATIONS
# ---------------------------------------------------------

# You can print simple checks to help spot patterns
print("\n--- Additional Observations ---")

# Check range per treatment
range_vals = treat_group["RelativeFitness"].apply(lambda x: x.max() - x.min())
print("\nRange of Relative Fitness per Treatment:")
print(range_vals)

# Check which treatment has highest average fitness
highest_avg = agg_results["mean"].idxmax()
print("\nTreatment with highest average fitness:", highest_avg)

# Check for outliers (simple rule: values > mean + 2*std)
outliers = df[df["RelativeFitness"] > df["RelativeFitness"].mean() + 2*df["RelativeFitness"].std()]
print("\nPossible Outliers:")
print(outliers)
