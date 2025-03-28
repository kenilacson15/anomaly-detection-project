import os
import logging
import pandas as pd
import numpy as np
import joblib  # for saving/loading models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve

# For explainability using SHAP
import shap

# Configure logging for detailed debugging output.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Define file paths
DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\feature_engineering\one_class_svm_feature\engineered_train_data_ocsvm.csv"
MODEL_RESULTS_DIR = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\model_results"
MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, "one_class_svm_model.joblib")
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_RESULTS_DIR, "hybrid_ensemble_model.joblib")
PREDICTIONS_PATH = os.path.join(MODEL_RESULTS_DIR, "train_predictions.csv")
SHAP_SUMMARY_PATH = os.path.join(MODEL_RESULTS_DIR, "shap_summary.png")
PR_CURVE_PATH = os.path.join(MODEL_RESULTS_DIR, "precision_recall_curve.png")

# Create the results directory if it doesn't exist.
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

def load_dataset(path):
    """
    Load the preprocessed CSV dataset into a pandas DataFrame.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        DataFrame: Loaded dataset.
    """
    logging.info("Loading dataset from %s", path)
    df = pd.read_csv(path)
    logging.info("Dataset shape: %s", df.shape)
    return df

def prepare_data(df):
    """
    Prepare the dataset by separating features from the target variable.
    
    For one-class SVM, we generally train only on 'normal' (non-faulty) instances.
    However, if the target column 'faulty' is present, we split it from the features.
    
    Args:
        df (DataFrame): The input DataFrame.
    
    Returns:
        X (DataFrame): Feature matrix (all columns except target).
        y (Series or None): Target variable if present, else None.
    """
    if 'faulty' in df.columns:
        y = df['faulty']
        X = df.drop(columns=['faulty'])
    else:
        y = None
        X = df.copy()
        
    logging.info("Features shape: %s", X.shape)
    if y is not None:
        logging.info("Target distribution:\n%s", y.value_counts())
    return X, y

def train_one_class_svm(X, nu=0.05, kernel='rbf', gamma='scale'):
    """
    Train a one-class SVM model for anomaly detection.
    
    Args:
        X (DataFrame): Feature matrix used for training.
        nu (float): Upper bound on training errors and lower bound on support vectors.
        kernel (str): Kernel type to use.
        gamma (str or float): Kernel coefficient.
    
    Returns:
        model (OneClassSVM): Trained one-class SVM model.
    """
    logging.info("Training One-Class SVM with nu=%s, kernel=%s, gamma=%s", nu, kernel, gamma)
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    model.fit(X)
    logging.info("One-Class SVM training completed.")
    return model

def train_isolation_forest(X, contamination=0.1, random_state=42):
    """
    Train an Isolation Forest model for anomaly detection.
    
    Args:
        X (DataFrame): Feature matrix.
        contamination (float): Proportion of anomalies in the data.
        random_state (int): Random seed.
    
    Returns:
        model (IsolationForest): Trained Isolation Forest model.
    """
    logging.info("Training Isolation Forest with contamination=%s", contamination)
    model = IsolationForest(contamination=contamination, random_state=random_state)
    model.fit(X)
    logging.info("Isolation Forest training completed.")
    return model

def tune_threshold(decision_scores, y_true, desired_precision=0.9):
    """
    Tune the decision threshold based on the precision-recall trade-off.
    
    Args:
        decision_scores (ndarray): Decision function scores from the One-Class SVM.
        y_true (ndarray): True binary labels (0 for normal, 1 for anomaly).
        desired_precision (float): The target precision we want to achieve.
        
    Returns:
        best_threshold (float): Decision threshold that meets the precision target.
    """
    # Since higher scores mean more inlier, lower scores indicate anomalies.
    precisions, recalls, thresholds = precision_recall_curve(y_true, -decision_scores)
    
    # Plot Precision-Recall vs. Threshold for visualization.
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision-Recall vs. Decision Threshold")
    plt.legend()
    plt.savefig(PR_CURVE_PATH)
    logging.info("Precision-Recall curve saved to %s", PR_CURVE_PATH)
    plt.close()

    # Find threshold where precision >= desired_precision.
    indices = np.where(precisions[:-1] >= desired_precision)[0]
    if len(indices) > 0:
        best_threshold = thresholds[indices[0]]
        logging.info("Selected threshold %.4f achieving precision %.4f", best_threshold, precisions[indices[0]])
    else:
        best_threshold = 0.0  # fallback if no threshold meets criteria
        logging.warning("No threshold found to achieve desired precision; using default threshold 0.0")
    return best_threshold

def apply_threshold(decision_scores, threshold):
    """
    Convert decision scores to binary predictions using the tuned threshold.
    
    Args:
        decision_scores (ndarray): Scores from the decision_function.
        threshold (float): The tuned threshold.
        
    Returns:
        predictions (ndarray): Binary predictions (0 for normal, 1 for anomaly).
    """
    # For One-Class SVM, normally: score > 0 -> normal; here we use -score compared with threshold.
    predictions = np.where(-decision_scores >= threshold, 1, 0)
    return predictions

def ensemble_predictions(pred_svm, pred_iforest):
    """
    Combine predictions from One-Class SVM and Isolation Forest using majority vote.
    
    Args:
        pred_svm (ndarray): Binary predictions from One-Class SVM.
        pred_iforest (ndarray): Binary predictions from Isolation Forest.
        
    Returns:
        ensemble_pred (ndarray): Final ensemble prediction.
    """
    # Majority vote: if both flag anomaly, label anomaly; otherwise, label normal.
    ensemble_pred = np.where((pred_svm + pred_iforest) >= 1, 1, 0)
    return ensemble_pred

def evaluate_model(y_true, predictions):
    """
    Evaluate predictions with confusion matrix and classification report.
    
    Args:
        y_true (ndarray): True binary labels.
        predictions (ndarray): Predicted binary labels.
        
    Returns:
        report (str): Classification report.
        conf_mat (ndarray): Confusion matrix.
    """
    conf_mat = confusion_matrix(y_true, predictions)
    report = classification_report(y_true, predictions, digits=4)
    logging.info("Confusion Matrix:\n%s", conf_mat)
    logging.info("Classification Report:\n%s", report)
    return report, conf_mat

def save_model(model, path):
    """
    Save a model to disk using joblib.
    
    Args:
        model: The trained model.
        path (str): File path for saving the model.
    """
    joblib.dump(model, path)
    logging.info("Model saved to %s", path)

def save_predictions(predictions, X, path):
    """
    Save predictions alongside feature data.
    
    Args:
        predictions (ndarray): Binary predictions.
        X (DataFrame): Feature matrix.
        path (str): Output CSV file path.
    """
    df_preds = X.copy()
    df_preds['predicted_faulty'] = predictions
    df_preds.to_csv(path, index=False)
    logging.info("Predictions saved to %s", path)

def visualize_data_distributions(X, y=None):
    """
    Create histograms for each feature to visualize distributions.
    
    If target labels are provided, overlay histograms for normal vs. anomaly.
    
    Args:
        X (DataFrame): Feature matrix.
        y (Series, optional): Target labels.
    """
    plt.figure(figsize=(16, 12))
    for i, column in enumerate(X.columns):
        plt.subplot(4, 4, i + 1)
        if y is not None:
            sns.histplot(X.loc[y == 0, column], color='blue', label='Normal', kde=True, stat="density", element="step", fill=False)
            sns.histplot(X.loc[y == 1, column], color='red', label='Anomaly', kde=True, stat="density", element="step", fill=False)
            plt.legend()
        else:
            sns.histplot(X[column], kde=True)
        plt.title(f"Distribution of {column}")
    plt.tight_layout()
    distribution_plot_path = os.path.join(MODEL_RESULTS_DIR, "feature_distributions.png")
    plt.savefig(distribution_plot_path)
    logging.info("Feature distribution plots saved to %s", distribution_plot_path)
    plt.close()

def visualize_pca(X, predictions):
    """
    Reduce the feature space to 2 dimensions using PCA and plot a scatter plot
    colored by predicted labels.
    
    Args:
        X (DataFrame): Feature matrix.
        predictions (ndarray): Binary predictions.
    """
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1],
                    hue=predictions, palette={0: "blue", 1: "red"}, legend='full')
    plt.title("PCA Projection Colored by Predicted Labels")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    pca_plot_path = os.path.join(MODEL_RESULTS_DIR, "pca_projection.png")
    plt.savefig(pca_plot_path)
    logging.info("PCA projection plot saved to %s", pca_plot_path)
    plt.close()

def visualize_confusion_matrix(conf_mat):
    """
    Create a heatmap of the confusion matrix.
    
    Args:
        conf_mat (ndarray): Confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    conf_mat_plot_path = os.path.join(MODEL_RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(conf_mat_plot_path)
    logging.info("Confusion matrix plot saved to %s", conf_mat_plot_path)
    plt.close()

def explain_false_positives(X, y_true, predictions, model, sample_size=50):
    """
    Use SHAP to explain why some normal instances are flagged as anomalies (false positives).
    
    Args:
        X (DataFrame): Feature matrix.
        y_true (Series): True binary labels.
        predictions (ndarray): Ensemble predictions.
        model (OneClassSVM): Trained One-Class SVM model.
        sample_size (int): Number of false positives to sample for explanation.
    """
    # Identify false positives: true normal (0) predicted as anomaly (1)
    fp_indices = X.index[(y_true == 0) & (predictions == 1)]
    if len(fp_indices) == 0:
        logging.info("No false positives to explain.")
        return
    sample_indices = fp_indices[:sample_size]
    X_fp = X.loc[sample_indices]
    
    # Use a subset of training data as background for SHAP KernelExplainer.
    background = X.sample(min(100, len(X)), random_state=42)
    
    # Create a prediction function using the SVM decision function.
    def svm_predict(data):
        return model.decision_function(data)
    
    explainer = shap.KernelExplainer(svm_predict, background, link="identity")
    shap_values = explainer.shap_values(X_fp, nsamples=100)
    
    # Plot SHAP summary for the false positives.
    shap.summary_plot(shap_values, X_fp, show=False)
    plt.tight_layout()
    plt.savefig(SHAP_SUMMARY_PATH)
    logging.info("SHAP summary plot saved to %s", SHAP_SUMMARY_PATH)
    plt.close()

def main():
    # Load and prepare dataset.
    df = load_dataset(DATA_PATH)
    X, y = prepare_data(df)
    
    # Visualize distributions.
    visualize_data_distributions(X, y)
    
    # For training, use only normal instances (label=0) if labels are provided.
    if y is not None:
        X_train = X[y == 0]
        logging.info("Training on normal instances only: %s samples", X_train.shape[0])
    else:
        X_train = X

    # Train One-Class SVM and Isolation Forest.
    svm_model = train_one_class_svm(X_train, nu=0.05, kernel='rbf', gamma='scale')
    if_model = train_isolation_forest(X, contamination=614/6138)  # contamination as ratio of anomalies
    
    # Compute decision scores from One-Class SVM for threshold tuning.
    decision_scores = svm_model.decision_function(X)
    # Tune threshold to achieve high precision (desired_precision can be adjusted).
    threshold = tune_threshold(decision_scores, y.values if y is not None else np.zeros(X.shape[0]), desired_precision=0.9)
    svm_predictions = apply_threshold(decision_scores, threshold)
    
    # Get predictions from Isolation Forest.
    # In IsolationForest: 1 -> normal, -1 -> anomaly; convert to binary.
    if_predictions = np.where(if_model.predict(X) == 1, 0, 1)
    
    # Ensemble: combine predictions from both models (majority vote).
    ensemble_pred = ensemble_predictions(svm_predictions, if_predictions)
    
    # Evaluate ensemble predictions if true labels are available.
    if y is not None:
        report, conf_mat = evaluate_model(y, ensemble_pred)
        visualize_confusion_matrix(conf_mat)
        logging.info("Ensemble Model Evaluation Report:\n%s", report)
    else:
        logging.info("No target labels provided; skipping evaluation.")
    
    # Visualize PCA projection of ensemble predictions.
    visualize_pca(X, ensemble_pred)
    
    # Save models and predictions.
    save_model(svm_model, MODEL_PATH)
    # Optionally save the isolation forest or ensemble (here we save ensemble predictions and models separately).
    save_model(if_model, os.path.join(MODEL_RESULTS_DIR, "isolation_forest_model.joblib"))
    save_predictions(ensemble_pred, X, PREDICTIONS_PATH)
    
    # Explain false positives using SHAP (if labels exist).
    if y is not None:
        explain_false_positives(X, y, ensemble_pred, svm_model)
    
    logging.info("Model training, evaluation, threshold tuning, ensemble, and explainability complete.")

if __name__ == '__main__':
    main()
