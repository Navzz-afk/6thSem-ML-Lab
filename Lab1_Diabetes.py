


from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes')

import pandas as pd

df = pd.read_csv(fn)

df.head()

print(df.isnull().sum())

print(f"Number of duplicate rows: {df.duplicated().sum()}")

import numpy as np

df = df.replace('?', np.nan)

print(df.isnull().sum())

print(df.describe())

print(df.dtypes)

print(df.columns.tolist())

df = pd.read_csv(fn)
df['CLASS'] = df['CLASS'].str.strip()
df = pd.get_dummies(df, columns=['Gender', 'CLASS'], drop_first=True)
df.head()

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numerical columns identified: {numerical_cols}")

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


for column in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[column])
    plt.title(f'Box Plot of {column}')
    plt.ylabel(column)
    plt.show()

for column in numerical_cols:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    num_outliers = len(outliers)

    print(f"Column: {column}")
    print(f"  Q1: {Q1:.2f}")
    print(f"  Q3: {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")
    print(f"  Number of Outliers: {num_outliers}")
    print("-----------------------------------------")

for column in numerical_cols:
    lower_bound_cap = df[column].quantile(0.05)
    upper_bound_cap = df[column].quantile(0.95)

    df[column] = df[column].clip(lower=lower_bound_cap, upper=upper_bound_cap)

print("DataFrame after capping outliers:")
df.head()

for column in ['workclass', 'occupation', 'native-country']:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]  # Get the first mode
        df[column].fillna(mode_value, inplace=True)
        print(f"Missing values in '{column}' filled with mode: '{mode_value}'")

print("\nNull values after imputation:")
print(df.isnull().sum())


for column in ['workclass', 'occupation', 'native-country']:
    if df[column].isnull().any():
        mode_value = df[column].mode()[0]  # Get the first mode
        df[column] = df[column].fillna(mode_value)
        print(f"Missing values in '{column}' filled with mode: '{mode_value}'")

print("\nNull values after imputation:")
print(df.isnull().sum())
