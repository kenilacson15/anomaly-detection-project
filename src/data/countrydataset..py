import pandas as pd

# File path
file_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\processed\equipment_anomaly_data_processed.csv"

# Load dataset
df = pd.read_csv(file_path)

# Ensure the column name is correct
location_column = "location"  # Update if needed

# Step 1: Check for missing values
missing_count = df[location_column].isnull().sum()
if missing_count > 0:
    print(f"Warning: {missing_count} missing values found in '{location_column}' column.")
    df = df.dropna(subset=[location_column])  # Drop rows with missing locations

# Step 2: Standardize locations (strip spaces, convert to title case)
df[location_column] = df[location_column].astype(str).str.strip().str.title()

# Step 3: Find unique locations
unique_locations = df[location_column].unique()

# Print results
print("\n✅ Locations Available in the Dataset:")
for loc in sorted(unique_locations):
    print("-", loc)
