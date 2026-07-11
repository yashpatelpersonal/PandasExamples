# ---------------------------------------------------------
# Cancer Data Analysis - Lab Assignment
# Author: Yash
# ---------------------------------------------------------

import pandas as pd

# ---------------------------------------------------------
# 1. DATA EXPLORATION
# ---------------------------------------------------------

# 1. Read dataset
df = pd.read_excel("Lab 7 - Cancer.xls")
#
print("#### Reading Data")
print(df)

# 2. Display data types
print("\n--- Data Types ---")
print(df.dtypes)

# 3. Summary statistics
print("\n--- Summary Statistics ---")
print(df.describe())

# 4. Group by class and doctor_name
grouped_df = df.groupby(['class', 'doctor_name'])
print("\n--- Grouped Data ---")
print(grouped_df.size())

# ---------------------------------------------------------
# 2. HANDLING MISSING VALUES
# ---------------------------------------------------------

# 5. Missing value summary
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 6. Drop rows with missing data
df_clean = df.dropna()

# 7. Count unique values per column
print("\n--- Unique Values ---")
print(df_clean.nunique())

# 8. Find duplicate patient_id values
dup_counts = df_clean['patient_id'].value_counts()
print("\n--- Duplicate patient_id counts ---")
print(dup_counts)

# print("\nMost frequent patient_id:", dup_counts.idxmax())

# ---------------------------------------------------------
# 3. FILTERING DATA
# ---------------------------------------------------------

# 9. Remove patients appearing more than twice
filtered_df = df_clean[df_clean['patient_id'].map(df_clean['patient_id'].value_counts()) <= 2]

print("\n--- Filtered Data (patient_id <= 2 occurrences) ---")
print(filtered_df)

# ---------------------------------------------------------
# 4. RESHAPING DATA
# ---------------------------------------------------------

# 10. Create categorical_df
categorical_df = filtered_df[['patient_id', 'doctor_name']].copy()
categorical_df['doctor_count'] = 1

# 11. Pivot table (one-hot style)
pivot_df = categorical_df.pivot_table(
    index='patient_id',
    columns='doctor_name',
    values='doctor_count',
    fill_value=0
)

# Safe droplevel
if isinstance(pivot_df.columns, pd.MultiIndex):
    pivot_df.columns = pivot_df.columns.droplevel(0)

print("\n--- One-Hot Encoded Doctor Data ---")
print(pivot_df)

# 14. Merge back with original filtered dataset
merged_df = filtered_df.merge(pivot_df, on='patient_id', how='left')

# 15. Display final combined DataFrame
print("\n--- Final Combined DataFrame ---")
print(merged_df)

# 16. Drop doctor_name column
merged_df = merged_df.drop(columns=['doctor_name'])

print("\n--- Final DF After Dropping doctor_name ---")
print(merged_df)

# ---------------------------------------------------------
# 5. ROW-WISE OPERATIONS
# ---------------------------------------------------------

# 17. Define celltypelabel function
def celltypelabel(x):
    if ((x['cell_size_uniformity'] > 5) &
            (x['cell_shape_uniformity'] > 5)):
        return 'normal'
    else:
        return 'abnormal'

# 18. Apply function to create new column
merged_df['cell_type_label'] = merged_df.apply(celltypelabel, axis=1)

print("\n--- Updated DataFrame with cell_type_label ---")
print(merged_df)
