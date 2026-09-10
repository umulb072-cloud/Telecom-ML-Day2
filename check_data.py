import pandas as pd

df = pd.read_csv("Data/feature_engineered_data.csv")

print("Target column values:")
print(df["Status"].value_counts())

print("\nPercentages:")
print(df["Status"].value_counts(normalize=True) * 100)