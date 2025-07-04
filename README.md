# Industrial Equipment 🖥️ Monitoring 🤲 Dataset

**Kaggle Dataset:** [Industrial Equipment Monitoring Dataset](https://www.kaggle.com/datasets/dnkumars/industrial-equipment-monitoring-dataset)

## 📂 About the Dataset

### 🔍 Overview
This dataset provides **simulated real-time sensor data** for industrial equipment, including **turbines, compressors, and pumps**. It captures key operational parameters such as **temperature, pressure, vibration, and humidity**. The dataset is structured with metadata, including **equipment type, location, and fault status**, making it suitable for anomaly detection and predictive maintenance applications.

### 📊 Features
- **`temperature`** (°C) – Sensor reading of equipment temperature
- **`pressure`** (bar) – Measured pressure level
- **`vibration`** – Normalized vibration reading
- **`humidity`** (%) – Relative humidity at equipment location
- **`equipment`** – Equipment type (Turbine, Compressor, Pump)
- **`location`** – Equipment location (e.g., city-based)
- **`faulty`** – Binary label (0 = Normal, 1 = Faulty)

### 🚀 Use Cases
- **Anomaly detection** in industrial processes
- **Predictive maintenance** to prevent failures
- **Performance trend analysis** of equipment

### 🔢 Dataset Statistics
- **Attributes**: 7
- **Example Equipment**: Turbine, Compressor, Pump
- **Example Locations**: Atlanta, Chicago, New York, San Francisco

---

## 🏗️ Data Preprocessing

### 1️⃣ Exploratory Data Analysis (EDA)
Performed initial analysis to understand dataset characteristics:
- Displayed dataset shape, data types, and basic statistics
- Identified missing values and duplicate records
- Visualized:
  - Histograms & boxplots for numerical features
  - Count plots for categorical features
  - Correlation heatmaps & pair plots for numerical features
  - Grouped statistics by **equipment type** and **location**
- Saved summary statistics as CSV files

### 2️⃣ Data Loading (`loader.py`)
- Implemented a **DataLoader** class for efficient data handling
- Checked file integrity and displayed dataset dimensions

### 3️⃣ Data Cleaning & Preprocessing (`preprocess.py`)
- **Handled missing values**:
  - Median imputation for numerical features
  - Mode imputation for categorical features
- **Removed duplicate records**
- **Outlier handling**:
  - Capped extreme values at **1st & 99th percentiles**
- **Data transformation**:
  - Converted categorical columns to appropriate types
  - Ensured `faulty` column is in numeric format

### 4️⃣ Data Splitting
- Performed **80-20 train-test split**, stratified on `faulty`
- Saved preprocessed data as `train_data.csv` and `test_data.csv`

### 📁 Output
- **Processed data & visualizations** stored in `data/preprocessing/`

---

## ⚙️ Feature Engineering
Feature engineering was **tailored per model** to maximize anomaly detection accuracy.

### 🔹 Autoencoder Feature Engineering
Optimized for **neural network-based anomaly detection**:
- **Standardized** numerical sensor readings (`StandardScaler`)
- **One-Hot Encoded** categorical features (`equipment`, `location`)
- **Handled missing values** (median for continuous, mode for categorical)
- **Pipeline-based transformation** for efficiency
📌 **Output**: `engineered_train_data_autoencoder.csv`

### 🔹 Isolation Forest Feature Engineering
Designed for **tree-based anomaly detection**:
- **Custom interaction features**:
  - `temp_pressure_ratio = temperature / (pressure + 1e-5)`
  - `temp_pressure_diff = temperature - pressure`
  - `vibration_humidity_product = vibration * humidity`
- **One-Hot Encoded** categorical features
- **Scaled numerical features** with `RobustScaler`
📌 **Output**: `engineered_train_data_isoforest.csv`

### 🔹 One-Class SVM Feature Engineering
Optimized for **hyperplane-based anomaly detection**:
- **Custom feature creation**:
  - `temp_pressure_ratio = temperature / (pressure + 1e-6)`
  - `vibration_humidity_product = vibration * humidity`
- **StandardScaler applied** to numerical features
- **One-Hot Encoding for categorical variables**
📌 **Output**: `engineered_train_data_ocsvm.csv`

---

## 🤖 Anomaly Detection System

### 1️⃣ Autoencoder Model (`autoencoder.py`)
- **Neural network architecture**: Encoder (64 → 32 neurons), bottleneck (12 neurons), decoder (32 → 64 neurons)
- **Robust optimization**:
  - Huber loss for resilience to anomalies
  - Gaussian noise injection (σ=0.01) to enhance generalization
- **Threshold tuning**: Set at **98th percentile** of reconstruction errors

### 2️⃣ Isolation Forest + LOF Ensemble
- **Isolation Forest**:
  - Contamination parameter set using estimated anomaly ratio
- **Local Outlier Factor (LOF)** added for **robust anomaly detection**
- **Majority voting ensemble** for final classification

### 3️⃣ One-Class SVM
- **RBF kernel with ν=0.05** for anomaly separation
- **Threshold optimized** via precision-recall curve

📌 **Output Artifacts**:
- Trained models (`.h5` / `.joblib`)
- Performance metrics, confusion matrices, and visualization plots

---

## 📊 Model Performance

| Model                    | F1-Score | Recall | Accuracy | ROC-AUC |
|--------------------------|----------|--------|----------|---------|
| **Autoencoder**         | 0.8742   | 0.89   | 96.5%    | 0.9842  |
| **Isolation Forest + LOF** | 0.67     | 0.84   | 92%      | 0.9625  |
| **One-Class SVM**       | 0.8793   | 0.9251 | 97.46%   | 0.9526  |
| **Ensemble Model**      | **0.8793** | **0.9251** | **97.46%** | - |

📌 **Insights**:
- **Autoencoder achieved the highest AUC-ROC (0.9842)**
- **One-Class SVM showed the highest recall (92.51%)**, making it preferable for high-risk anomaly detection
- **Ensemble learning improved model robustness**, minimizing false negatives

📊 **Confusion Matrix Saved To:** `data/model_results/confusion_matrix.png`

---

## 📝 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

```
Copyright 2025 [Ken Ira]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

🚀 **For more details, check out the full dataset on [Kaggle](https://www.kaggle.com/datasets/dnkumars/industrial-equipment-monitoring-dataset)**.