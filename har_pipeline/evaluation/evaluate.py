import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from har_pipeline.scripts.load_har_dataset import load_and_preprocess_dataset
from har_pipeline.scripts.utils import load_and_compile_model
import argparse
import yaml
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved HAR model")
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--model_path', default=None, help='Path to saved model or checkpoint (h5 or checkpoint prefix)')
    parser.add_argument('--custom_loop_model', action='store_true', help='Use custom training loop model format')
    return parser.parse_args()


def extratct_labels(test_ds: tf.data.Dataset, convert_to_single_labels=True) -> np.ndarray:
    """
    Extracts true labels from the dataset.
    Args:
        dataset: tf.data.Dataset object
    Returns:
        numpy array of true labels
    """
    # Extract true labels from test_ds
    y_true_list = []
    for _, labels in test_ds:  # Iterates through batches of (features, labels)
        y_true_list.append(labels.numpy())  # Converts TensorFlow tensor to NumPy array
    y_true = np.concatenate(y_true_list, axis=0)  # Concatenates all batch labels into one array

    if convert_to_single_labels:
        # If your labels are one-hot encoded, convert them back to single labels
        # The UCI HAR dataset typically has integer labels 1-6, which are often
        # one-hot encoded during TF.data preprocessing.
        if y_true.ndim > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
    
    return y_true


def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Read configuration parameters using .get() for safer access and default values
    training_config = config.get('training', {})
    model_dir = training_config.get('model_dir', 'models')
    learning_rate = training_config.get('learning_rate', 0.001) # Needed for compiling

    # Load data
    _, test_ds = load_and_preprocess_dataset(config)

    # Load model
    # --- Determine the model path to load ---
    target_model_path = args.model_path
    if target_model_path is None:
        if args.custom_loop_model:
            target_model_path = os.path.join(model_dir, 'best_model_custom.keras')
        else:
            target_model_path = os.path.join(model_dir, 'best_model.keras')

    # --- Load and Compile the Model ---
    model = load_and_compile_model(
        model_path=target_model_path,
        is_custom_loop_model=args.custom_loop_model,
        test_ds=test_ds,
        learning_rate=learning_rate
    )

    # Evaluate model on test set
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

    # Predict class labels for classification report
    y_pred_prob = model.predict(test_ds)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = extratct_labels(test_ds, convert_to_single_labels=True)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
