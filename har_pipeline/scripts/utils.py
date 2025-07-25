import tensorflow as tf
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def split_batched_dataset(batched_dataset: tf.data.Dataset, validation_percentage: float) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Splits an already shuffled and batched tf.data.Dataset into training and validation sets.

    Args:
        batched_dataset: A tf.data.Dataset that has already been shuffled and batched.
                         Its cardinality (number of batches) must be known.
        validation_percentage: A float between 0.0 and 1.0 indicating the
                               percentage of batches to use for the validation set.

    Returns:
        A tuple containing two tf.data.Dataset objects: (train_dataset, val_dataset).

    Raises:
        ValueError: If validation_percentage is out of range, or if the dataset
                    cardinality is unknown or infinite.
    """
    if not (0.0 <= validation_percentage <= 1.0):
        raise ValueError("validation_percentage must be between 0.0 and 1.0")

    # Get the total number of batches in the dataset
    total_batches = tf.data.experimental.cardinality(batched_dataset).numpy()

    if total_batches == tf.data.UNKNOWN_CARDINALITY:
        raise ValueError(
            "Cannot split dataset: Cardinality is UNKNOWN_CARDINALITY. "
            "Ensure your dataset's size can be determined (e.g., not from a generator without a specified output_shapes/output_types, or not using .repeat() without a count)."
        )
    if total_batches == tf.data.INFINITE_CARDINALITY:
        raise ValueError(
            "Cannot split an INFINITE_CARDINALITY dataset using this method. "
            "For infinite datasets, manage validation by setting validation_steps in model.fit()."
        )

    # Calculate the number of batches for validation
    val_batches = int(total_batches * validation_percentage)
    train_batches = total_batches - val_batches

    # Ensure there's at least one batch for both train and validation if possible
    if train_batches == 0 and total_batches > 0:
        print(f"Warning: validation_percentage {validation_percentage*100:.1f}% results in 0 training batches. Adjusting to 1 training batch.")
        train_batches = 1
        val_batches = total_batches - 1
    elif val_batches == 0 and total_batches > 0 and validation_percentage > 0:
        print(f"Warning: validation_percentage {validation_percentage*100:.1f}% results in 0 validation batches. Adjusting to 1 validation batch.")
        val_batches = 1
        train_batches = total_batches - 1
    
    if total_batches == 0:
        print("Warning: Input dataset is empty. Returning two empty datasets.")
        return batched_dataset, batched_dataset # Return empty datasets

    print(f"Total batches: {total_batches}")
    print(f"Validation batches: {val_batches}")
    print(f"Training batches: {train_batches}")

    # Split the dataset
    train_dataset = batched_dataset.take(train_batches)
    val_dataset = batched_dataset.skip(train_batches).take(val_batches)

    return train_dataset, val_dataset

def get_dataset_shapes(dataset: tf.data.Dataset) -> tuple[tuple, int]:
    """
    Infers the input shape for a model and the number of classes from a tf.data.Dataset.

    This function assumes the dataset yields tuples of (features, labels),
    where features are suitable for a model input (e.g., (sequence_length, num_features))
    and labels are one-hot encoded (e.g., (num_classes,)).

    Args:
        dataset: A tf.data.Dataset object. It should contain at least one element
                 (batch) to infer shapes correctly.

    Returns:
        A tuple containing:
        - input_shape (tuple): The shape of a single data sample (e.g., (128, 9) for HAR data),
                               excluding the batch dimension.
        - num_classes (int): The number of output classes.

    Raises:
        ValueError: If the dataset is empty, or if the structure of the dataset
                    elements is not as expected (e.g., not (features, labels)).
    """
    try:
        # Get a single element (batch) from the dataset.
        # .take(1) creates a new dataset with only the first element.
        # .get_single_element() extracts that element as a tuple of tensors.
        # This is efficient as it doesn't iterate through the whole dataset.
        features_batch, labels_batch = dataset.take(1).get_single_element()
    except tf.errors.OutOfRangeError:
        raise ValueError("The provided dataset is empty. Cannot infer shapes from an empty dataset.")
    except Exception as e:
        raise ValueError(
            f"Could not get a single element from the dataset. "
            f"Ensure it yields (features, labels) tuples. Error: {e}"
        )

    # Infer input_shape:
    # The shape of features_batch will be (batch_size, sequence_length, num_features).
    # The model's input_shape should exclude the batch_size dimension.
    input_shape = features_batch.shape[1:]

    # Infer num_classes:
    # The shape of labels_batch will typically be (batch_size, num_classes) if one-hot encoded.
    # The second dimension gives the number of classes.
    if labels_batch.shape.rank < 2:
        raise ValueError(
            f"Labels batch has shape {labels_batch.shape}. "
            "Expected one-hot encoded labels with rank >= 2 (e.g., (batch_size, num_classes)). "
            "If labels are not one-hot encoded, you might need to adjust this function."
        )
    num_classes = labels_batch.shape[1]

    return input_shape, num_classes