
import pandas as pd

# 1. Load the dataset
df = pd.read_csv("data/creditcard.csv")

# 2. Display first five rows
print("\n========== FIRST FIVE ROWS ==========")
print(df.head())

# 3. Display dataset information
print("\n========== DATASET INFORMATION ==========")
df.info()

# 4. Display dataset dimensions
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# 5. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().sum())

# 6. Count genuine and fraudulent transactions
print("\n========== CLASS DISTRIBUTION ==========")
print(df["Class"].value_counts())

# 7. Calculate percentages
print("\n========== CLASS PERCENTAGES ==========")
print(df["Class"].value_counts(normalize=True) * 100)

# 8. Display statistical information
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
