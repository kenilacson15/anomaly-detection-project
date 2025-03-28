import pandas as pd
import os

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads the raw dataset efficiently without preprocessing."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ File not found: {self.file_path}")

        try:
            df = pd.read_csv(self.file_path)
            print(f"✅ Successfully loaded dataset with {df.shape[0]} rows & {df.shape[1]} columns.")
            return df

        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    file_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\raw\equipment_anomaly_data.csv"
    loader = DataLoader(file_path)
    df = loader.load_data()
    print(df.head())  # Display first 5 rows
