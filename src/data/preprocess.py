# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set paths for input raw data and output preprocessed data
raw_data_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\raw\equipment_anomaly_data.csv"
output_dir = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# 1. Load the dataset
df = pd.read_csv(raw_data_path)
print("Initial shape:", df.shape)

# 2. Initial Data Check
# Display the first few rows
print("\nFirst 5 rows:")
print(df.head())

# Data info: data types and non-null counts
print("\nData Information:")
print(df.info())

# Summary statistics for numeric features
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicate rows
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed. New shape:", df.shape)

# 3. Data Cleaning and Preprocessing
# Handling missing values (if any exist)
# For demonstration, we'll fill missing numeric values with the median and categorical with mode
for col in df.columns:
    if df[col].dtype in ['float64', 'int64']:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Filled missing numeric values in '{col}' with median:", median_val)
    else:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Filled missing categorical values in '{col}' with mode:", mode_val)

# Optional: Convert categorical columns to category dtype (or encode later during modeling)
categorical_columns = ['equipment', 'location', 'faulty']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Optional: Basic outlier handling (example: cap numeric features at 1st and 99th percentiles)
numeric_features = ['temperature', 'pressure', 'vibration', 'humidity']
for col in numeric_features:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    print(f"Capped values in '{col}' to the range [{lower_bound}, {upper_bound}]")

# Ensure target variable ('faulty') is numeric (if not, convert it)
if df['faulty'].dtype.name != 'int64':
    df['faulty'] = df['faulty'].astype(int)

# 4. Split the Data into Training and Testing Sets
# Use stratified split if the target is imbalanced
X = df.drop('faulty', axis=1)
y = df['faulty']

# For this demonstration, use a 80-20 train-test split with stratification on 'faulty'
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Combine features and target back for saving
train_data = X_train.copy()
train_data['faulty'] = y_train

test_data = X_test.copy()
test_data['faulty'] = y_test

print("\nTraining set shape:", train_data.shape)
print("Testing set shape:", test_data.shape)

# 5. Save the Cleaned and Split Data to CSV files
train_output_path = os.path.join(output_dir, 'train_data.csv')
test_output_path = os.path.join(output_dir, 'test_data.csv')

train_data.to_csv(train_output_path, index=False)
test_data.to_csv(test_output_path, index=False)

print("\nPreprocessing complete. Cleaned training and testing datasets are saved at:")
print(train_output_path)
print(test_output_path)
