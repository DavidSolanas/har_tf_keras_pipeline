# HAR TensorFlow Keras Pipeline

This repository provides a complete, modular, and production-ready pipeline for the Human Activity Recognition (HAR) dataset using TensorFlow and `tf.keras`.

## Features

- Modular code: data loading, model, training separated
- Configurable hyperparameters via YAML
- Reproducibility with random seed setting
- Efficient data pipeline using `tf.data`
- Residual CNN model architecture for time series classification
- Both standard and custom training loops included
- Evaluation script with detailed metrics
- Automated testing with `pytest`
- Code style enforcement with `flake8`
- Continuous Integration (CI) with GitHub Actions:
  - Runs linting, testing, and coverage reports automatically on pushes and PRs

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
```
---

## Dataset

Download the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and place it in:

```
data/UCI HAR Dataset/
```

The folder should contain the `train/` and `test/` subdirectories with the raw `.txt` files like `X_train.txt`, `y_train.txt`, etc.

To preprocess the dataset:

```bash
python preprocess_har.py
```

This script will normalize the data and optionally save the processed files as `.npy` for reuse in training scripts.

---

## Running the Training

### ✅ Standard Training Loop

```bash
python timeseries/train_standard_loop.py
```

### 🔁 Custom Training Loop (tf.GradientTape)

```bash
python timeseries/train_custom_loop.py
```

> Configuration can be adjusted via `config/config.yaml`

---

## Evaluation

Run the evaluation script to test the model and print classification metrics:

```bash
python timeseries/evaluate.py
```

---

## Project Structure

```
har-pipeline/
├── config/
│   └── config.yaml              # Training configuration
├── data/
│   └── UCI HAR Dataset/         # Contains the original dataset from UCI
├── models/
│   └── residual_cnn.py          # Model architecture
├── scripts/
│   |── preprocess_har.py        # Data prep script
|   └── utils.py                 # Utils script
├── timeseries/
│   ├── train_standard_loop.py   # Training with model.fit()
│   ├── train_custom_loop.py     # Training with GradientTape
│   └── evaluate.py              # Final evaluation
├── tests/
│   └── test_model.py            # Unit tests for model
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions CI config
├── .flake8                      # Code style rules
├── requirements.txt
├── requirements-test.txt
└── README.md
```

---

## Configuration

The training is configurable via a YAML file:

```yaml
# config/config.yaml
epochs: 30
batch_size: 64
learning_rate: 0.001
dropout_rate: 0.5
model:
  filters: 64
  kernel_size: 3
```

---

## Testing and Linting

Run unit tests with coverage:

```bash
pytest --cov=timeseries tests/
```

Run style checks:

```bash
flake8 .
```

---

## CI/CD with GitHub Actions

The CI pipeline (`.github/workflows/ci.yml`) automatically runs:

- `flake8` for code style
- `pytest` for unit tests
- Coverage reporting

This ensures consistency and reliability in pull requests and main branches.

---

## .flake8

Basic config to enforce PEP8 standards:

```ini
[flake8]
max-line-length = 88
ignore = E203, W503
exclude = .git,__pycache__,.venv
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- UCI Machine Learning Repository: [HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- TensorFlow team for powerful tools and documentation
