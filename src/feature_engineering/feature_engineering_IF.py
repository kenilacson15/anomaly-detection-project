import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def load_data(input_path):
    """
    Load raw training data from CSV.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"Data loaded successfully from {input_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")
    return df

def feature_engineering(df):
    """
    Perform feature engineering for anomaly detection using Isolation Forest.
    """

    # 1. Create interaction features
    # Temperature/Pressure ratio: Can capture relationship between temperature and pressure.
    df['temp_pressure_ratio'] = df['temperature'] / (df['pressure'] + 1e-5)
    
    # Temperature - Pressure difference: Highlights absolute differences.
    df['temp_pressure_diff'] = df['temperature'] - df['pressure']
    
    # Vibration-Humidity product: A possible indicator of equipment stress.
    df['vibration_humidity_product'] = df['vibration'] * df['humidity']

    # 2. One-hot encode categorical variables ('equipment' and 'location')
    categorical_cols = ['equipment', 'location']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # 3. (Optional) Scaling numerical features: While Isolation Forest is tree based
    # and generally does not require scaling, applying a RobustScaler can help
    # if you plan on combining or thresholding anomaly scores.
    # List all numerical columns (exclude target 'faulty' if you want to keep it aside)
    numerical_cols = [col for col in df_encoded.columns if col not in ['faulty']]
    
    scaler = RobustScaler()
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    
    return df_encoded

def save_data(df, output_dir, output_filename="train_data_engineered.csv"):
    """
    Save the engineered dataframe to the output directory.
    """
    # Create output directory if it does not exist.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    output_path = os.path.join(output_dir, output_filename)
    try:
        df.to_csv(output_path, index=False)
        print(f"Engineered data saved successfully to {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save engineered data: {e}")

def main():
    # Define input and output paths
    input_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\preprocess_data\train_data.csv"
    output_dir = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\feature_engineering"
    
    # Load raw data
    df = load_data(input_path)
    
    # Feature engineering
    df_engineered = feature_engineering(df)
    
    # Save the engineered dataset
    save_data(df_engineered, output_dir)

if __name__ == "__main__":
    main()
