import tensorflow as tf
import yaml
import os
from har_pipeline.models.residual_cnn import build_model
from har_pipeline.scripts.load_har_dataset import load_and_preprocess_dataset
from har_pipeline.scripts.utils import set_seed, split_batched_dataset, get_dataset_shapes
import argparse

def parse_args():
    """
    Parse command line arguments for training script.
    """
    parser = argparse.ArgumentParser(description="Train HAR model")
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Read configuration parameters using .get() for safer access and default values
    training_config = config.get('training', {})

    seed = training_config.get('seed', 42)
    model_dir = training_config.get('model_dir', 'models')
    logs_dir = training_config.get('logs_dir', 'logs')
    epochs = training_config.get('epochs', 100)
    validation_percentage = training_config.get('validation_percentage', 0.2)
    learning_rate = training_config.get('learning_rate', 0.001)
    patience = training_config.get('patience', 10) # Default patience for early stopping
    
    # Set seeds for reproducibility
    set_seed(seed)

    # Load data
    train_ds, test_ds = load_and_preprocess_dataset(config)

    # Create tf.data.Dataset pipelines
    train_ds, val_ds = split_batched_dataset(train_ds, validation_percentage)
    
    # Get dataset shapes AFTER splitting and batching if batching affects shape
    input_shape, num_classes = get_dataset_shapes(train_ds) 

    # Always build the model from scratch here.
    # tf.keras.callbacks.BackupAndRestore will restore its state if resuming.
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    # Compile model
    # This must be done *before* adding BackupAndRestore, as BackupAndRestore
    # needs to save the optimizer state, which is part of compilation.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy', # Assuming one-hot encoded labels
        metrics=['accuracy']
    )

    # Prepare callbacks
    os.makedirs(model_dir, exist_ok=True)
    
    # Define a directory for BackupAndRestore to save/load temporary checkpoints
    backup_dir = os.path.join(model_dir, 'backup_restore_temp')
    os.makedirs(backup_dir, exist_ok=True) # Ensure backup directory exists

    callbacks = [
        # Callback to save the best model based on validation accuracy
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        # Early stopping callback
        tf.keras.callbacks.EarlyStopping(
            patience=patience, # Use patience from config
            restore_best_weights=False, # Do not Restore weights from the best epoch found by ES, save the best model separately
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        # TensorBoard callback
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir),
        
        # --- NEW: BackupAndRestore callback for seamless resume ---
        # This callback manages saving/restoring the full training state (model, optimizer, epoch).
        # It detects if a previous run was interrupted and automatically resumes from the last state.
        tf.keras.callbacks.BackupAndRestore(
            backup_dir=backup_dir # Directory for temporary backup files
        )
    ]

    # Train
    # With BackupAndRestore, you do NOT set initial_epoch manually.
    # It handles that internally based on the restored state.
    print(f"Starting training for {epochs} epochs (will resume if backup exists in {backup_dir})...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # The 'final_model.keras' will be the state of the model after early stopping (if it occurred)
    # or after all epochs complete. If restore_best_weights is True in EarlyStopping,
    # this will be the best model seen during training, so it might be identical to best_model.keras.
    print(f"Training finished. Final model saved to: {os.path.join(model_dir, 'final_model.keras')}")
    model.save(os.path.join(model_dir, 'final_model.keras'))


    # Evaluate on test set
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(test_ds)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()