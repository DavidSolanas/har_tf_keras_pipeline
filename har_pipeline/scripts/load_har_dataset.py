import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import yaml
from pathlib import Path
import tensorflow as tf


def load_signal_file(filepath):
    """Loads a signal file (e.g., 'body_acc_x_train.txt')."""
    return pd.read_csv(filepath, sep=r'\s+', header=None).values

def load_signals(signal_list, data_path, label):
    """Loads all inertial signals for a set (train/test).
    Args:
        signal_list: List of signal names to load (e.g., ['total_acc_x', 'total_acc_y', ...])
        data_path: Path to the train or test directory
        label: 'train' or 'test' to specify which set to load
    Returns:
        A numpy array of shape (num_samples, TIME_STEPS, FEATURES) containing the stacked signals.
    """
    # List to store the matrices for each signal (e.g., 7352, 128)
    all_signals = []
    for signal_name in signal_list:
        file_path = os.path.join(data_path, 'Inertial Signals', f'{signal_name}_{label}.txt')
        signal_data = load_signal_file(file_path)
        all_signals.append(signal_data)
    
    # Stack the signals to get (num_samples, TIME_STEPS, FEATURES)
    # For example, for train: (7352, 128, 9)
    # The order is rearranged so that feature columns are on the last axis.
    stacked_signals = np.stack(all_signals, axis=-1)
    return stacked_signals

def load_dataset(dataset_path, signals: list):
    """Loads the complete HAR dataset.
    Args:
        dataset_path: Path to the dataset directory containing 'train' and 'test' subdirectories.
        signals: List of variable names corresponding to the dataset signals.
    Returns:
        X_train_signals: Training signals (numpy array)
        y_train: Training labels (numpy array)
        X_test_signals: Test signals (numpy array)
        y_test: Test labels (numpy array)
        activity_labels: List of activity names
    """
    print(f"Loading dataset from: {dataset_path}")

    # Names of the train and test subdirectories
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    # Label files
    train_labels_path = os.path.join(train_path, 'y_train.txt')
    test_labels_path = os.path.join(test_path, 'y_test.txt')

    # Activity names file
    activity_labels_path = os.path.join(dataset_path, 'activity_labels.txt')

    # Load labels
    y_train = load_signal_file(train_labels_path)
    y_test = load_signal_file(test_labels_path)

    # Load activity names
    activity_labels_df = pd.read_csv(activity_labels_path, sep=r'\s+', header=None)
    activity_labels = activity_labels_df[1].tolist()
    print(f"Detected activities: {activity_labels}")

    # Load training signals
    print("Loading training signals...")
    X_train_signals = load_signals(signals, train_path, 'train')
    print(f"Shape of X_train_signals: {X_train_signals.shape}")

    # Load test signals
    print("Loading test signals...")
    X_test_signals = load_signals(signals, test_path, 'test')
    print(f"Shape of X_test_signals: {X_test_signals.shape}")
    
    return X_train_signals, y_train, X_test_signals, y_test, activity_labels

def preprocess_data(X_train: np.ndarray, y_train: np.ndarray, 
                    X_test: np.ndarray, y_test: np.ndarray,
                    time_steps: int, features: int):
    """
    Normalizes the input data and encodes the labels.
    Args:
        X_train: Training data (numpy array)
        y_train: Training labels (numpy array)
        X_test: Test data (numpy array)
        y_test: Test labels (numpy array)
        time_steps: Number of time steps (integer)
        features: Number of features (integer)
    Returns:
        X_train_processed: Normalized training data
        y_train_encoded: One-hot encoded training labels
        X_test_processed: Normalized test data
        y_test_encoded: One-hot encoded test labels
        num_classes: Number of unique classes in the labels
    """
    print("Normalizing X data...")
    # Normalize each feature across all time windows.
    # Data is in the format (samples, time_steps, features).
    # We need to temporarily flatten it for the scaler.
    
    # Flatten data so StandardScaler treats it as a list of features per window
    # From (samples, time_steps, features) to (samples * time_steps, features)
    n_train_samples, _, _ = X_train.shape
    n_test_samples, _, _ = X_test.shape

    X_train_flat = X_train.reshape(-1, features)
    X_test_flat = X_test.reshape(-1, features)

    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)

    # Reshape the data back to (samples, time_steps, features)
    X_train_processed = X_train_scaled_flat.reshape(n_train_samples, time_steps, features)
    X_test_processed = X_test_scaled_flat.reshape(n_test_samples, time_steps, features)

    print("Encoding Y labels (One-Hot Encoding)...")
    # Labels are in the range 1-6. Keras expects 0-indexed.
    # Subtract 1 so labels are 0, 1, 2, 3, 4, 5.
    y_train_processed = y_train - 1
    y_test_processed = y_test - 1

    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_train_processed)
    y_test_encoded = encoder.transform(y_test_processed)
    
    num_classes = y_train_encoded.shape[1]
    print(f"Number of classes: {num_classes}")

    return X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, num_classes

def create_tf_dataset(features, labels, batch_size, shuffle=True):
    """
    Create a tf.data.Dataset pipeline from numpy arrays.

    Args:
        features: np.ndarray, features
        labels: np.ndarray, labels
        batch_size: int, batch size for training
        shuffle: bool, whether to shuffle the dataset

    Returns:
        tf.data.Dataset object
    """
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def load_and_preprocess_dataset(config) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Loads the HAR UCI dataset, preprocess it and creates a tf.data.Dataset for training
    and testing.

    Returns:
        ds_train: Train preprocessed dataset (tf.data.Dataset)
        ds_test: Test preprocessed dataset (tf.data.Dataset)
    """
    # --- Configuration and Dataset Loading ---
    # Always resolve relative to root of repo
    repo_path = Path(__file__).resolve().parents[2]

    dataset_path = repo_path / Path(config["dataset"]["path"])
    time_steps = config["dataset"]["time_steps"]
    signals = config["dataset"]["signals"]

    # Number of features per time step
    features = len(signals) 

    # Batch size for dataset loader
    batch_size = config["training"]["batch_size"]

    # 1. Load the dataset
    X_train, y_train, X_test, y_test, activity_labels = load_dataset(dataset_path, signals)

    # 2. Preprocess the data
    X_train_processed, y_train_encoded, X_test_processed, y_test_encoded, num_classes = \
        preprocess_data(X_train, y_train, X_test, y_test, time_steps, features)
    
    ds_train = create_tf_dataset(X_train_processed, y_train_encoded, batch_size)
    ds_test = create_tf_dataset(X_test_processed, y_test_encoded, batch_size, shuffle=False)

    return ds_train, ds_test