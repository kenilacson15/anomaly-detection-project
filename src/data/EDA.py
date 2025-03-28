# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths for input raw data and output EDA results
raw_data_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\raw\equipment_anomaly_data.csv"
output_dir = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing"

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Load the dataset
df = pd.read_csv(raw_data_path)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nData Types and Non-null Counts:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())

# Summary statistics for numeric features
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# --- Visualization Section ---

# 1. Histograms for numeric features
numeric_features = ['temperature', 'pressure', 'vibration', 'humidity']
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'numeric_features_histograms.png'))
plt.close()

# 2. Boxplots to check for outliers in numeric features
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'numeric_features_boxplots.png'))
plt.close()

# 3. Count plots for categorical features: equipment, location, and faulty
categorical_features = ['equipment', 'location', 'faulty']
plt.figure(figsize=(18, 5))
for i, col in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count plot of {col}')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'categorical_features_countplots.png'))
plt.close()

# 4. Correlation heatmap for numeric features
plt.figure(figsize=(8, 6))
corr_matrix = df[numeric_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

# 5. Pairplot for a more detailed look at pairwise relationships (sample a subset if dataset is large)
sns.pairplot(df, vars=numeric_features, hue='faulty', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Numeric Features Colored by Faulty", y=1.02)
plt.savefig(os.path.join(output_dir, 'pairplot_numeric_features.png'))
plt.close()

# Additional EDA: Group statistics by equipment and location
equipment_group = df.groupby('equipment')[numeric_features].agg(['mean', 'std', 'min', 'max'])
location_group = df.groupby('location')[numeric_features].agg(['mean', 'std', 'min', 'max'])
print("\nGrouped Statistics by Equipment:")
print(equipment_group)
print("\nGrouped Statistics by Location:")
print(location_group)

# Save grouped statistics to CSV files
equipment_group.to_csv(os.path.join(output_dir, 'grouped_stats_by_equipment.csv'))
location_group.to_csv(os.path.join(output_dir, 'grouped_stats_by_location.csv'))

print(f"\nEDA results and plots have been saved to: {output_dir}")
