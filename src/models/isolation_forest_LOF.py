import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix, 
                             precision_recall_curve, average_precision_score, roc_curve)
from sklearn.model_selection import GridSearchCV, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib
import mlflow

# -----------------------------------
# Custom Wrapper for Isolation Forest
# -----------------------------------
class IsolationForestWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.1, random_state=42, n_jobs=-1, n_estimators=100, max_features=1.0):
        self.contamination = contamination
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.max_features = max_features
        
    def fit(self, X, y=None):
        self.model_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            max_features=self.max_features
        )
        self.model_.fit(X)
        # Define classes manually: 0 = normal, 1 = anomaly.
        self.classes_ = np.array([0, 1])
        return self
    
    def decision_function(self, X):
        # Flip sign so that higher values indicate more anomalous
        return -self.model_.decision_function(X)
    
    def predict(self, X):
        preds = self.model_.predict(X)
        return np.where(preds == 1, 0, 1)

# -----------------------------------
# Custom Wrapper for Local Outlier Factor
# -----------------------------------
class LOFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, contamination=0.1, n_neighbors=20, novelty=True):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.novelty = novelty  # Must be True to allow scoring on new data
       
    def fit(self, X, y=None):
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, 
            contamination=self.contamination,
            novelty=self.novelty
        )
        self.model_.fit(X)
        self.classes_ = np.array([0, 1])
        return self
    
    def decision_function(self, X):
        # LOF returns lower scores for outliers; flip sign so that higher means more anomalous
        return -self.model_.decision_function(X)
    
    def predict(self, X):
        preds = self.model_.predict(X)
        return np.where(preds == 1, 0, 1)

# -----------------------------------
# Ensemble Wrapper Combining IF and LOF
# -----------------------------------
class EnsembleAnomalyDetector(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 isoforest_params=None, 
                 lof_params=None, 
                 weight_iso=0.5, 
                 weight_lof=0.5):
        # Set default parameters if none provided
        if isoforest_params is None:
            isoforest_params = {
                'contamination': 0.1, 
                'random_state': 42, 
                'n_jobs': -1, 
                'n_estimators': 100, 
                'max_features': 1.0
            }
        if lof_params is None:
            lof_params = {
                'contamination': 0.1, 
                'n_neighbors': 20, 
                'novelty': True
            }
        # Store parameters for proper cloning and reproducibility
        self.isoforest_params = isoforest_params
        self.lof_params = lof_params
        self.weight_iso = weight_iso
        self.weight_lof = weight_lof
        
        # Instantiate the underlying models
        self.isoforest = IsolationForestWrapper(**self.isoforest_params)
        self.lof = LOFWrapper(**self.lof_params)
    
    def fit(self, X, y=None):
        self.isoforest.fit(X)
        self.lof.fit(X)
        self.classes_ = np.array([0, 1])
        return self
    
    def decision_function(self, X):
        # Combine scores using a weighted average
        score_iso = self.isoforest.decision_function(X)
        score_lof = self.lof.decision_function(X)
        ensemble_score = self.weight_iso * score_iso + self.weight_lof * score_lof
        return ensemble_score
    
    def predict(self, X):
        # Default decision using 0 threshold (to be refined via threshold optimization)
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, 0)

# -----------------------------------
# 1. Configuration & Data Loading
# -----------------------------------
data_path = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\feature_engineering\engineered_features_IF\train_data_engineered.csv"
output_dir = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\model_results"
os.makedirs(output_dir, exist_ok=True)

print("Loading data...")
data = pd.read_csv(data_path)
print("Data loaded successfully with shape:", data.shape)

# -----------------------------------
# 2. Feature Selection and Label Extraction
# -----------------------------------
feature_columns = [
    'temperature', 'pressure', 'vibration', 'humidity', 
    'temp_pressure_ratio', 'temp_pressure_diff', 'vibration_humidity_product', 
    'equipment_Pump', 'equipment_Turbine', 
    'location_Chicago', 'location_Houston', 'location_New York', 'location_San Francisco'
]
X = data[feature_columns]
y = data['faulty']

# -----------------------------------
# 3. Determine Contamination Parameter
# -----------------------------------
contamination = y.mean()
print(f"Calculated contamination rate: {contamination:.4f}")

# -----------------------------------
# 4. Build Preprocessing & Ensemble Model Pipeline
# -----------------------------------
# Define parameters for underlying models
isoforest_params = {'contamination': contamination, 'random_state': 42, 'n_jobs': -1, 'n_estimators': 100, 'max_features': 0.8}
lof_params = {'contamination': contamination, 'n_neighbors': 20, 'novelty': True}

ensemble_model = EnsembleAnomalyDetector(
    isoforest_params=isoforest_params,
    lof_params=lof_params,
    weight_iso=0.5,
    weight_lof=0.5
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ensemble', ensemble_model)
])

# -----------------------------------
# 5. Hyperparameter Tuning via Grid Search
# -----------------------------------
# Tune the weights between IF and LOF.
param_grid = {
    'ensemble__weight_iso': [0.4, 0.5, 0.6],
    'ensemble__weight_lof': [0.6, 0.5, 0.4],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def custom_scorer(estimator, X, y):
    scores = estimator.decision_function(X)
    return roc_auc_score(y, scores)

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    scoring=custom_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X, y)
print("Best parameters found:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# -----------------------------------
# 6. Prediction, Anomaly Scoring & Threshold Optimization
# -----------------------------------
# Obtain cross-validated decision scores.
y_scores = cross_val_predict(
    best_model, X, y,
    method='decision_function',
    cv=cv,
    n_jobs=-1
)
# Higher scores indicate more anomalous instances.

# Determine optimal threshold via precision-recall curve
precision, recall, thresholds = precision_recall_curve(y, y_scores)

# Optimize using F1-score:
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
optimal_idx_f1 = np.argmax(f1_scores)
optimal_threshold_f1 = thresholds[optimal_idx_f1]
print(f"Optimal anomaly threshold determined (F1): {optimal_threshold_f1:.4f}")

# Optimize using F2-score (beta=2 for higher recall emphasis):
beta = 2
f2_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)
optimal_idx_f2 = np.argmax(f2_scores)
optimal_threshold_f2 = thresholds[optimal_idx_f2]
print(f"Optimal anomaly threshold determined (F2): {optimal_threshold_f2:.4f}")

# Choose the F2 threshold if recall is critical.
optimal_threshold = optimal_threshold_f2

# Generate predictions with both the default and threshold-optimized decision rule.
data['anomaly_score'] = best_model.decision_function(X)
data['predicted_faulty'] = best_model.predict(X)
data['predicted_faulty_opt'] = (data['anomaly_score'] >= optimal_threshold).astype(int)

# -----------------------------------
# 7. Evaluation
# -----------------------------------
print("\nClassification Report (default predictions):")
print(classification_report(y, data['predicted_faulty']))

print("\nClassification Report (optimized threshold predictions):")
print(classification_report(y, data['predicted_faulty_opt']))

roc_auc = roc_auc_score(y, data['anomaly_score'])
print("ROC-AUC Score:", roc_auc)

print("Confusion Matrix (optimized threshold):")
print(confusion_matrix(y, data['predicted_faulty_opt']))

# -----------------------------------
# 8. Save the Model and Artifacts
# -----------------------------------
model_path = os.path.join(output_dir, "ensemble_anomaly_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

results_path = os.path.join(output_dir, "train_data_with_predictions.csv")
data.to_csv(results_path, index=False)
print(f"Prediction results saved to {results_path}")

# -----------------------------------
# 9. Generate and Save Diagnostic Plots
# -----------------------------------
def generate_evaluation_plots(y_true, scores, threshold, save_path):
    plt.figure(figsize=(20, 5))
    
    # Subplot 1: Precision-Recall Curve
    plt.subplot(1, 4, 1)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    plt.plot(rec, prec, marker='.')
    plt.title(f'PR Curve (AP={average_precision_score(y_true, scores):.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # Subplot 2: Anomaly Score Distribution
    plt.subplot(1, 4, 2)
    plt.hist(scores, bins=50, alpha=0.7)
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold={threshold:.2f}')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Subplot 3: Confusion Matrix Heatmap
    plt.subplot(1, 4, 3)
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Normal','Anomalous'])
    plt.yticks(tick_marks, ['Normal','Anomalous'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Subplot 4: ROC Curve
    plt.subplot(1, 4, 4)
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, marker='.')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.title(f'ROC Curve (AUC={roc_auc_score(y_true, scores):.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
plot_path = os.path.join(output_dir, "evaluation_plots.png")
generate_evaluation_plots(y, data['anomaly_score'], optimal_threshold, plot_path)
print(f"Diagnostic plots saved to {plot_path}")

def generate_additional_plots(y_true, scores, save_path):
    plt.figure(figsize=(12, 6))
    
    # Boxplot: Anomaly scores by true label
    plt.subplot(1, 2, 1)
    data_for_box = pd.DataFrame({'Anomaly Score': scores, 'True Label': y_true})
    data_for_box.boxplot(column='Anomaly Score', by='True Label')
    plt.title('Anomaly Score Distribution by Label')
    plt.suptitle('')
    plt.xlabel('True Label (0: Normal, 1: Anomalous)')
    plt.ylabel('Anomaly Score')
    
    # Scatter Plot: Anomaly score vs. sample index colored by true label
    plt.subplot(1, 2, 2)
    indices = np.arange(len(scores))
    plt.scatter(indices, scores, c=y_true, cmap='coolwarm', alpha=0.7)
    plt.colorbar(label='True Label (0: Normal, 1: Anomalous)')
    plt.title('Anomaly Scores Across Samples')
    plt.xlabel('Sample Index')
    plt.ylabel('Anomaly Score')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
additional_plot_path = os.path.join(output_dir, "additional_plots.png")
generate_additional_plots(y, data['anomaly_score'], additional_plot_path)
print(f"Additional diagnostic plots saved to {additional_plot_path}")

print("\nEnsemble anomaly detection pipeline completed successfully.")
