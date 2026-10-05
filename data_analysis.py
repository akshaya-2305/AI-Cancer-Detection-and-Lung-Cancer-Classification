import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

# Load the Breast Cancer Wisconsin Diagnostic dataset
data = fetch_ucirepo(id=17)

# Separate features and target
X = data.data.features
y = data.data.targets

print("Dataset loaded successfully!")

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nFirst 5 rows of features:")
print(X.head())

print("\nFirst 5 target values:")
print(y.head())

print("\n" + "="*50)
print("DATASET INFORMATION")
print("="*50)

# Column names
print("\nColumn names:")
print(X.columns.tolist())

# Data types
print("\nData types:")
print(X.dtypes)

# Missing values
print("\nMissing values:")
print(X.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(X.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(X.describe())