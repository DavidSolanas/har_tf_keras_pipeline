# tests/conftest.py
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture
def mock_config():
    return {
        "dataset": {
            "time_steps": 128,
            "features": 9,
            "path": "tests/fake_data",
            "signals": ["signal_x_acc", "signal_y_acc", "signal_z_acc"]
        },
        "training": {
            "batch_size": 16
        }
    }

@pytest.fixture
def input_shape():
    return (128, 9)

@pytest.fixture
def number_of_classes():
    return 6

@pytest.fixture
def dummy_data(input_shape):
    X = np.random.randn(10, *input_shape).astype(np.float32)
    y = tf.keras.utils.to_categorical(np.random.randint(0, 6, size=(10,)), num_classes=6)
    return X, y

@pytest.fixture
def mock_raw_data():
    # Generate fake dataset similar to UCI HAR
    X_train = np.random.randn(64, 128, 9)
    X_test = np.random.randn(32, 128, 9)
    y_train = np.random.randint(1, 7, (64, 1))
    y_test = np.random.randint(1, 7, (32, 1))
    return X_train, y_train, X_test, y_test