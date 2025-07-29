import numpy as np
import pandas as pd
from unittest.mock import patch, mock_open
from har_pipeline.scripts import load_har_dataset
import tensorflow as tf

def test_load_signal_file_reads_correct_shape():
    fake_data = "1 2 3\n4 5 6\n"
    with patch("builtins.open", mock_open(read_data=fake_data)):
        with patch("pandas.read_csv") as mock_read_csv:
            # Return a DataFrame instead of a numpy array
            mock_read_csv.return_value = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
            result = load_har_dataset.load_signal_file("dummy_path.txt")
            assert result.shape == (2, 3)

def test_load_signals_stacks_correctly():
    dummy_signal = np.ones((5, 128))  # 5 samples, 128 timesteps
    with patch("har_pipeline.scripts.load_har_dataset.load_signal_file", return_value=dummy_signal):
        signals = ["signal_x", "signal_y", "signal_z"]
        stacked = load_har_dataset.load_signals(signals, "dummy_path", "train")
        assert stacked.shape == (5, 128, 3)  # 3 signals

def test_preprocess_data_shapes(mock_raw_data, number_of_classes):
    X_train, y_train, X_test, y_test = mock_raw_data
    
    X_train_p, y_train_e, X_test_p, y_test_e, num_classes = load_har_dataset.preprocess_data(
        X_train, y_train, X_test, y_test, 128, 9
    )
    
    assert num_classes == number_of_classes
    assert X_train_p.shape == X_train.shape
    assert X_test_p.shape == X_test.shape
    assert y_train_e.shape == (X_train.shape[0], num_classes)
    assert y_test_e.shape == (X_test.shape[0], num_classes)

def test_create_tf_dataset(input_shape, number_of_classes):
    features = np.random.rand(20, *input_shape)
    labels = np.eye(6)[np.random.choice(number_of_classes, 20)]
    
    dataset = load_har_dataset.create_tf_dataset(features, labels, batch_size=4)
    batch = next(iter(dataset))
    
    X_batch, y_batch = batch
    assert X_batch.shape[0] == 4
    assert y_batch.shape[0] == 4
    for batch in dataset:
        x, y = batch
        assert x.shape[1:] == input_shape
        assert y.shape[1:] == (number_of_classes,)


def test_load_and_preprocess_dataset_full(mocker, mock_config, mock_raw_data):
    X_train, y_train, X_test, y_test = mock_raw_data
    signals = mock_config["dataset"]["signals"]
    time_steps = mock_config["dataset"]["time_steps"]
    n_signals = len(signals)
    samples_train = X_train.shape[0]
    samples_test = X_test.shape[0]

    # Create fake train/test signals: one 2D array (samples, time_steps) per signal
    fake_train_signals = [np.random.randn(samples_train, time_steps) for _ in range(n_signals)]
    fake_test_signals = [np.random.randn(samples_test, time_steps) for _ in range(n_signals)]

    # Patch os.path.join to join path parts as strings separated by '/' (to avoid PosixPath issues)
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(map(str, args)))

    # Patch pandas.read_csv to return the appropriate fake data based on filepath string
    def fake_read_csv(filepath, sep=None, header=None):
        path_str = str(filepath)
        if "y_train.txt" in path_str:
            return pd.DataFrame(y_train)
        elif "y_test.txt" in path_str:
            return pd.DataFrame(y_test)
        elif "activity_labels.txt" in path_str:
            # Return consistent activity labels for 6 classes
            return pd.DataFrame({
                0: list(range(1, 7)),
                1: ["WALKING", "SITTING", "LAYING", "STANDING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]
            })
        elif "train" in path_str:
            # Extract signal name and return corresponding fake train signal
            signal_name = path_str.split("/")[-1].replace(".txt", "").replace("_train", "")
            idx = signals.index(signal_name)
            return pd.DataFrame(fake_train_signals[idx])
        elif "test" in path_str:
            # Extract signal name and return corresponding fake test signal
            signal_name = path_str.split("/")[-1].replace(".txt", "").replace("_test", "")
            idx = signals.index(signal_name)
            return pd.DataFrame(fake_test_signals[idx])
        else:
            raise ValueError(f"Unexpected file path in test: {path_str}")

    mocker.patch("pandas.read_csv", side_effect=fake_read_csv)

    # Run the full load + preprocess pipeline
    train_ds, test_ds = load_har_dataset.load_and_preprocess_dataset(mock_config)

    # Assert datasets are created and are tf.data.Dataset objects
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(test_ds, tf.data.Dataset)

    # Check one batch from train_ds: features shape (batch_size, time_steps, features), labels one-hot encoded
    x_batch, y_batch = next(iter(train_ds))
    assert x_batch.shape == (mock_config["training"]["batch_size"], time_steps, n_signals)
    assert y_batch.shape == (mock_config["training"]["batch_size"], 6)  # 6 classes

    # Similarly, check one batch from test_ds (shuffled=False)
    x_batch_test, y_batch_test = next(iter(test_ds))
    assert x_batch_test.shape == (mock_config["training"]["batch_size"], time_steps, n_signals)
    assert y_batch_test.shape == (mock_config["training"]["batch_size"], 6)