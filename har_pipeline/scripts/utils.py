import tensorflow as tf
import numpy as np
import random
import os
from har_pipeline.models.residual_cnn import build_model

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

def load_and_compile_model(model_path, is_custom_loop_model, test_ds, learning_rate, loss_fn='categorical_crossentropy') -> tf.keras.Model:
    """
    Loads a TensorFlow Keras model and compiles it based on its origin (custom loop or model.fit).

    Args:
        model_path (str): Path to the saved model file (.keras/.h5) or checkpoint directory/prefix.
        is_custom_loop_model (bool): True if the model was trained with the custom loop script,
                                     False if trained with model.fit().
        test_ds (tf.data.Dataset): The test dataset, used to infer input_shape and num_classes
                                   if the model architecture needs to be rebuilt.
        learning_rate (float): The learning rate, used for compiling the model if it needs to be compiled.
        loss_fn (str): The loss function to use for compiling the model. Default is 'categorical_crossentropy'.

    Returns:
        tf.keras.Model: The loaded and compiled TensorFlow Keras model.

    Raises:
        FileNotFoundError: If a specified checkpoint or model file is not found.
        RuntimeError: If the model cannot be loaded or built.
        ValueError: If the model_path format is unknown.
    """
    model = None
    input_shape, num_classes = get_dataset_shapes(test_ds) # Get shapes early

    if is_custom_loop_model:
        # Scenario 1: Model saved from custom training loop
        # This can be either 'best_model_custom.keras' or a CheckpointManager directory
        
        model = build_model(input_shape=input_shape, num_classes=num_classes) # Always rebuild for custom loop

        if model_path.endswith('.keras') or model_path.endswith('.h5'):
            # It's the best_model_custom.keras file from the custom loop
            try:
                # Try loading as a full Keras model first
                loaded_model = tf.keras.models.load_model(model_path)
                # If loaded successfully, use this model instead of the freshly built one
                model = loaded_model
                print(f"Full Keras model '{os.path.basename(model_path)}' (from custom loop's best save) loaded.")
            except Exception as e:
                print(f"Could not load full Keras model directly '{os.path.basename(model_path)}': {e}. Attempting to load weights only...")
                model.load_weights(model_path) # Load weights into the newly built model
                print("Model weights loaded into rebuilt architecture.")
        else:
            # Assume it's a CheckpointManager's directory/prefix
            checkpoint_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else model_path
            latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if latest_checkpoint:
                checkpoint = tf.train.Checkpoint(model=model)
                checkpoint.restore(latest_checkpoint).expect_partial()
                print(f"Model weights restored from custom loop checkpoint: {latest_checkpoint}")
            else:
                raise FileNotFoundError(f"No custom loop checkpoint found in {checkpoint_dir}")
                
        # For custom loop models, compilation is always needed if not loaded as a full .keras model
        # or even if loaded as full .keras but you want to re-confirm compile params.
        print("Compiling model (from custom loop) for evaluation...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss_fn,
            metrics=['accuracy']
        )

    else:
        # Scenario 2: Model saved from model.fit() script (default behavior)
        # These are typically full Keras models (.keras or .h5)
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"Full Keras model '{os.path.basename(model_path)}' (from model.fit() save) loaded.")
            # If loaded as a full model, it should already be compiled.
        except Exception as e:
            # If `model.load_model` fails, it's usually because it's not a full saved model
            # or there are custom objects it doesn't know about.
            # In this case, we try to rebuild and load weights as a fallback.
            print(f"Could not load full Keras model directly '{os.path.basename(model_path)}': {e}. Attempting to rebuild and load weights...")
            model = build_model(input_shape=input_shape, num_classes=num_classes)
            try:
                model.load_weights(model_path)
                print("Model weights loaded into rebuilt architecture.")
                # If weights were loaded, it needs to be compiled
                print("Compiling model (from model.fit() weights) for evaluation...")
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
            except Exception as e_weights:
                raise RuntimeError(f"Failed to load model from {model_path} even as weights. "
                                   f"Error: {e_weights}. "
                                   f"Ensure it's a valid Keras model/weights file "
                                   f"or use --custom_loop_model flag if appropriate.")
    
    if model is None:
        raise RuntimeError("Model could not be loaded or built.")
    
    # Final check for compilation state in case of unexpected scenarios (e.g., loaded weights without explicit compile path)
    if not hasattr(model, 'optimizer') or model.optimizer is None:
        print("Final check: Model appears not compiled. Compiling for evaluation...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss= loss_fn,
            metrics=['accuracy']
        )
            
    return model