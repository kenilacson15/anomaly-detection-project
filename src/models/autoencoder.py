import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, GaussianNoise
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import tensorflow as tf
import datetime

# ---------------------------------------------------------------------------
# Set random seeds for reproducibility
# ---------------------------------------------------------------------------
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
DATA_PATH = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\preprocessing\feature_engineering\engineered_train_data_autoencoder\engineered_train_data_autoencoder.csv"
MODEL_OUTPUT_DIR = r"C:\Users\Ken Ira Talingting\Desktop\anomaly-detection-project\data\model_results"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Data Loading and Preprocessing
# ---------------------------------------------------------------------------
def load_data(path):
    """Load the preprocessed training dataset from a CSV file."""
    logging.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    """
    Separate features and target. This dataset is already feature-engineered.
    
    For autoencoder training, we use only normal samples (faulty == 0).
    
    Returns:
        X: Features as a DataFrame (all columns except 'faulty').
        y: Target as a Series.
        X_normal: Subset of features corresponding to normal samples.
    """
    # Separate features and target
    X = df.drop('faulty', axis=1)
    y = df['faulty'].astype(int)
    # For training autoencoder, use only normal samples (faulty == 0)
    X_normal = X[y == 0].copy()
    logging.info("Data split: %d normal samples and %d anomalies", X_normal.shape[0], (y == 1).sum())
    return X, y, X_normal

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------
def build_autoencoder(input_dim, encoding_dim=12, dropout_rate=0.15, l1_reg=1e-6, l2_reg=1e-4):
    """
    Build and compile an optimized autoencoder.
    
    Architecture:
      - Input Noise: Adds Gaussian noise to improve robustness.
      - Encoder: Two Dense layers (64, then 32 neurons) with BatchNormalization, LeakyReLU, and Dropout.
      - Bottleneck: Dense layer (encoding_dim neurons) with LeakyReLU.
      - Decoder: Mirror the encoder with two Dense layers (32, then 64 neurons) with BatchNormalization, LeakyReLU, and Dropout.
      - Output: Reconstruction layer with linear activation.
    
    Uses Huber loss (more robust to outliers) and Adam optimizer.
    """
    input_layer = Input(shape=(input_dim,), name='input_layer')
    
    # Input noise for regularization
    x = GaussianNoise(0.01, name='input_noise')(input_layer)
    
    # Encoder
    x = Dense(64, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name='encoder_dense1')(x)
    x = BatchNormalization(name='encoder_bn1')(x)
    x = LeakyReLU(alpha=0.1, name='encoder_leakyrelu1')(x)
    x = Dropout(dropout_rate, name='encoder_dropout1')(x)
    
    x = Dense(32, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name='encoder_dense2')(x)
    x = BatchNormalization(name='encoder_bn2')(x)
    x = LeakyReLU(alpha=0.1, name='encoder_leakyrelu2')(x)
    x = Dropout(dropout_rate, name='encoder_dropout2')(x)
    
    # Bottleneck
    bottleneck = Dense(encoding_dim, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name='bottleneck_dense')(x)
    bottleneck = LeakyReLU(alpha=0.1, name='bottleneck_leakyrelu')(bottleneck)
    
    # Decoder (mirror encoder)
    x = Dense(32, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name='decoder_dense1')(bottleneck)
    x = BatchNormalization(name='decoder_bn1')(x)
    x = LeakyReLU(alpha=0.1, name='decoder_leakyrelu1')(x)
    x = Dropout(dropout_rate, name='decoder_dropout1')(x)
    
    x = Dense(64, kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg), name='decoder_dense2')(x)
    x = BatchNormalization(name='decoder_bn2')(x)
    x = LeakyReLU(alpha=0.1, name='decoder_leakyrelu2')(x)
    x = Dropout(dropout_rate, name='decoder_dropout2')(x)
    
    # Reconstruction layer
    output_layer = Dense(input_dim, activation='linear', name='output_layer')(x)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name='autoencoder')
    autoencoder.compile(optimizer=Adam(), loss=Huber())
    logging.info("Autoencoder built with input_dim=%d, encoding_dim=%d", input_dim, encoding_dim)
    return autoencoder

# ---------------------------------------------------------------------------
# Training and Callbacks
# ---------------------------------------------------------------------------
def train_autoencoder(autoencoder, X_train, epochs=100, batch_size=32):
    """
    Train the autoencoder with early stopping, learning rate reduction, model checkpointing,
    and TensorBoard logging.
    """
    # Setup TensorBoard log directory
    log_dir = os.path.join(MODEL_OUTPUT_DIR, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(filepath=os.path.join(MODEL_OUTPUT_DIR, 'best_autoencoder.h5'),
                        monitor='loss', save_best_only=True, verbose=1),
        TensorBoard(log_dir=log_dir)
    ]
    
    logging.info("Starting training for %d epochs with batch size %d", epochs, batch_size)
    history = autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size,
                              shuffle=True, callbacks=callbacks, verbose=1)
    logging.info("Training completed")
    return history

# ---------------------------------------------------------------------------
# Evaluation and Metrics
# ---------------------------------------------------------------------------
def evaluate_model(autoencoder, X, y, threshold_quantile=0.98):
    """
    Evaluate the autoencoder using reconstruction error.
    The anomaly threshold is determined by the specified quantile of errors on normal data.
    
    Returns:
        metrics: Dictionary containing ROC-AUC, F1-score, and threshold.
        errors: Reconstruction error for each sample.
        y_pred: Binary predictions (0: normal, 1: anomaly).
    """
    reconstructions = autoencoder.predict(X)
    # Use Mean Squared Error per sample as reconstruction error
    errors = np.mean(np.square(X - reconstructions), axis=1)
    threshold = np.quantile(errors[y == 0], threshold_quantile)
    logging.info("Anomaly detection threshold set at: %.4f", threshold)
    
    y_pred = (errors > threshold).astype(int)
    roc_auc = roc_auc_score(y, errors)
    f1 = f1_score(y, y_pred)
    
    logging.info("Evaluation Metrics - ROC-AUC: %.4f, F1-score: %.4f", roc_auc, f1)
    print("\nClassification Report:\n", classification_report(y, y_pred))
    
    metrics = {'roc_auc': roc_auc, 'f1_score': f1, 'threshold': threshold}
    return metrics, errors, y_pred

# ---------------------------------------------------------------------------
# Visualization and Saving
# ---------------------------------------------------------------------------
def plot_error_distribution(errors, threshold, output_dir):
    """
    Plot the distribution of reconstruction errors and mark the anomaly threshold.
    The plot is saved to the specified output directory.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, label='Reconstruction Error')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plot_path = os.path.join(output_dir, 'reconstruction_error_distribution.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info("Reconstruction error distribution plot saved to %s", plot_path)

def save_model(autoencoder, output_dir):
    """
    Save the trained autoencoder model to the specified directory.
    """
    model_path = os.path.join(output_dir, 'autoencoder_model.h5')
    autoencoder.save(model_path)
    logging.info("Model saved to %s", model_path)

def save_performance_report(metrics, output_dir):
    """
    Save the performance report (ROC-AUC, F1-score, threshold) to a text file.
    """
    report_path = os.path.join(output_dir, 'model_performance_report.txt')
    with open(report_path, 'w') as f:
        f.write("Autoencoder Model Performance Report\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n")
        f.write(f"Anomaly Detection Threshold: {metrics['threshold']:.4f}\n")
    logging.info("Model performance report saved to %s", report_path)

# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------
def main():
    # Load and preprocess the data
    df = load_data(DATA_PATH)
    X, y, X_normal = preprocess_data(df)
    
    # Convert to numpy arrays with proper dtype (float32)
    X_all = X.values.astype(np.float32)
    X_normal_np = X_normal.values.astype(np.float32)
    
    # Build and train the autoencoder on normal samples only
    input_dim = X_all.shape[1]
    autoencoder = build_autoencoder(input_dim=input_dim, encoding_dim=12, dropout_rate=0.15)
    history = train_autoencoder(autoencoder, X_normal_np, epochs=100, batch_size=32)
    
    # Evaluate the model on the full dataset
    metrics, errors, y_pred = evaluate_model(autoencoder, X_all, y, threshold_quantile=0.98)
    plot_error_distribution(errors, metrics['threshold'], MODEL_OUTPUT_DIR)
    
    # Save the final model and performance report
    save_model(autoencoder, MODEL_OUTPUT_DIR)
    save_performance_report(metrics, MODEL_OUTPUT_DIR)

if __name__ == '__main__':
    main()
