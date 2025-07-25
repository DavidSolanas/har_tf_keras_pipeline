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

    # Set seeds for reproducibility
    set_seed(config['training']['seed'])

    # Read configuration parameters
    model_dir = config['training']['model_dir']
    logs_dir = config['training']['logs_dir']
    epochs = config['training']['epochs']
    validation_percentage = config['training']['validation_percentage']
    learning_rate = config['training']['learning_rate']

    # Load data
    train_ds, test_ds = load_and_preprocess_dataset(config)

    # Create tf.data.Dataset pipelines
    train_ds, val_ds = split_batched_dataset(train_ds, validation_percentage)
    input_shape, num_classes = get_dataset_shapes(train_ds)

    # Build model
    model = build_model(input_shape=input_shape, num_classes=num_classes)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Prepare callbacks
    os.makedirs(model_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.keras'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
    ]

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save final model
    model.save(os.path.join(model_dir, 'final_model.keras'))

    # Evaluate on test set
    loss, accuracy = model.evaluate(test_ds)
    print(f"Validation Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
