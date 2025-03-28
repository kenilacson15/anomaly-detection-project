import os
import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# File paths
INPUT_DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\preprocess_data\train_data.csv"
OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\feature_engineering"
OUTPUT_DATA_PATH = os.path.join(OUTPUT_DIR, "engineered_train_data_autoencoder.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data(path):
    """
    Load the raw CSV data into a pandas DataFrame.
    """
    logging.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    return df

def feature_engineering(df):
    """
    Perform feature engineering:
      - Handle missing values in continuous and categorical features.
      - Standard scale continuous features.
      - One-hot encode categorical features.
      - Return a new DataFrame with engineered features and the target variable.
    """
    # Define columns
    continuous_cols = ['temperature', 'pressure', 'vibration', 'humidity']
    categorical_cols = ['equipment', 'location']
    target_col = 'faulty'
    
    # Fill missing values for continuous features with the median
    for col in continuous_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info("Filled missing values in %s with median: %f", col, median_val)
    
    # Fill missing values for categorical features with the mode
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logging.info("Filled missing values in %s with mode: %s", col, mode_val)
    
    # Create a preprocessing pipeline using ColumnTransformer:
    # - StandardScaler for continuous features
    # - OneHotEncoder for categorical features (using sparse_output=False for newer scikit-learn versions)
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), continuous_cols),
        ('cat', OneHotEncoder(sparse_output=False), categorical_cols)
    ])
    
    # Fit and transform the relevant columns
    X_transformed = preprocessor.fit_transform(df[continuous_cols + categorical_cols])
    
    # Construct feature names:
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = continuous_cols + list(cat_feature_names)
    
    # Create a DataFrame from the transformed features
    df_features = pd.DataFrame(X_transformed, columns=feature_names)
    
    # Add the target variable back to the DataFrame
    df_features[target_col] = df[target_col].values
    
    logging.info("Feature engineering completed with %d features.", len(feature_names))
    return df_features

def save_engineered_data(df, output_path):
    """
    Save the engineered DataFrame as a CSV file.
    """
    df.to_csv(output_path, index=False)
    logging.info("Engineered dataset saved to: %s", output_path)

def main():
    # Load raw data
    df = load_data(INPUT_DATA_PATH)
    
    # Apply feature engineering
    engineered_df = feature_engineering(df)
    
    # Save the engineered dataset for use in training your autoencoder
    save_engineered_data(engineered_df, OUTPUT_DATA_PATH)

if __name__ == '__main__':
    main()
