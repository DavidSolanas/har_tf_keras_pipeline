import tensorflow as tf
import yaml
import os
import time
from har_pipeline.models.residual_cnn import build_model
from har_pipeline.scripts.load_har_dataset import load_and_preprocess_dataset
from har_pipeline.scripts.utils import set_seed, split_batched_dataset, get_dataset_shapes
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train HAR model with custom training loop")
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the last saved checkpoint')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Read configuration parameters using .get() for safer access and default values
    # It's good practice to provide sensible defaults if a parameter might be omitted from config.yaml
    training_config = config.get('training', {}) # Safely get the 'training' section, default to empty dict
    seed = training_config.get('seed', 42) # Default seed for reproducibility
    model_dir = training_config.get('model_dir', 'models') # Default model directory
    logs_dir = training_config.get('logs_dir', 'logs')   # Default logs directory
    epochs = training_config.get('epochs', 100)           # Default number of training epochs
    validation_percentage = training_config.get('validation_percentage', 0.2) # Default validation split
    learning_rate = training_config.get('learning_rate', 0.001) # Default learning rate for Adam optimizer
    patience = training_config.get('patience', 10)         # Default patience for early stopping
    checkpoint_save_freq = training_config.get('checkpoint_save_freq', 1) # Default to saving every epoch for resume
    max_checkpoints = training_config.get('max_checkpoints', 2) # Default to keeping last 2 checkpoints

    # Set seeds for reproducibility
    set_seed(seed)

    # Load data
    train_ds, _ = load_and_preprocess_dataset(config)

    # Create tf.data.Dataset pipelines
    train_ds, val_ds = split_batched_dataset(train_ds, validation_percentage)
    input_shape, num_classes = get_dataset_shapes(train_ds)

    # Build model
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    # Define optimizer and loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_acc_metric = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    # Prepare checkpoint directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Initialize Checkpoint object
    # Including epoch, best_val_acc, and wait as tf.Variable allows them to be saved/restored
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model,
                                     epoch=tf.Variable(0), # Current epoch
                                     best_val_acc=tf.Variable(0.0), # Best validation accuracy seen so far
                                     wait=tf.Variable(0)) # Patience counter for early stopping

    # Initialize CheckpointManager to manage resume checkpoints
    # It will keep the 2 most recent checkpoints for resuming training
    manager = tf.train.CheckpointManager(checkpoint, directory=model_dir, max_to_keep=max_checkpoints)

    initial_epoch = 0
    best_val_acc_initial = 0.0 # Use a temporary variable for initial best_val_acc
    wait_initial = 0 # Use a temporary variable for initial wait

    # --- Resume Training Logic ---
    if args.resume_training:
        if manager.latest_checkpoint:
            print(f"Resuming training from checkpoint: {manager.latest_checkpoint}")
            # Restore the checkpoint. This will populate model, optimizer, and our tf.Variables.
            checkpoint.restore(manager.latest_checkpoint).expect_partial()
            initial_epoch = checkpoint.epoch.numpy()
            best_val_acc_initial = checkpoint.best_val_acc.numpy()
            wait_initial = checkpoint.wait.numpy()
            print(f"Resuming from epoch {initial_epoch + 1} with best_val_acc={best_val_acc_initial:.4f}, wait={wait_initial}")
        else:
            print("No checkpoint found to resume from. Starting training from scratch.")
    # --- End Resume Training Logic ---

    # Assign initial values for the loop
    best_val_acc = best_val_acc_initial
    wait = wait_initial

    # TensorBoard writers
    # TensorBoard will automatically append to existing logs if the log directory is the same.
    train_log_dir = os.path.join(logs_dir, 'train')
    val_log_dir = os.path.join(logs_dir, 'val')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = loss_fn(y, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss_metric.update_state(loss)
        train_acc_metric.update_state(y, logits)

    @tf.function
    def val_step(x, y):
        logits = model(x, training=False)
        loss = loss_fn(y, logits)
        val_loss_metric.update_state(loss)
        val_acc_metric.update_state(y, logits)

    for epoch in range(initial_epoch, epochs):
        # Update the epoch variable in the checkpoint object
        checkpoint.epoch.assign(epoch + 1) 

        start_time = time.time()

        # Reset metrics for the new epoch
        train_loss_metric.reset_state()
        train_acc_metric.reset_state()
        val_loss_metric.reset_state()
        val_acc_metric.reset_state()

        # Training loop over batches
        for x_batch, y_batch in train_ds:
            train_step(x_batch, y_batch)

        # Validation loop over batches
        for x_val, y_val_batch in val_ds:
            val_step(x_val, y_val_batch)

        train_loss = train_loss_metric.result()
        train_acc = train_acc_metric.result()
        val_loss = val_loss_metric.result()
        val_acc = val_acc_metric.result()

        # TensorBoard logging
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)
        with val_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar('accuracy', val_acc, step=epoch)

        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Time: {time.time() - start_time:.2f}s")

        # Early stopping & saving the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            # Update the tf.Variable in the checkpoint for persistence
            checkpoint.best_val_acc.assign(best_val_acc)
            checkpoint.wait.assign(wait)
            # Save the best model as a full Keras model for easy deployment
            # This is separate from the CheckpointManager's resume checkpoints
            model.save(os.path.join(model_dir, 'best_model_custom.keras'))
            print(f"Saved best model at epoch {epoch+1} with Val Acc: {best_val_acc:.4f}")
        else:
            wait += 1
            # Update the tf.Variable in the checkpoint for persistence
            checkpoint.wait.assign(wait)
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Save a checkpoint for resuming training using CheckpointManager
        # This will keep the latest N checkpoints (max_to_keep=2)
        if (epoch + 1) % checkpoint_save_freq == 0 or (epoch + 1) == epochs:
            save_path = manager.save()
            print(f"Saved resume checkpoint for epoch {epoch+1} at {save_path}")

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    # Save final model (the model's state at the very end of training)
    model.save(os.path.join(model_dir, 'final_model_custom.keras'))

if __name__ == "__main__":
    main()