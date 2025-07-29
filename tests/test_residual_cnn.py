import pytest
import tensorflow as tf
from har_pipeline.models.residual_cnn import build_model, residual_block

def test_residual_block_output_shape():
    input_tensor = tf.random.normal((32, 128, 64))
    output_tensor = residual_block(input_tensor, filters=64, kernel_size=3)
    assert output_tensor.shape == input_tensor.shape

def test_build_model_structure():
    model = build_model(input_shape=(128, 9), num_classes=6)
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None, 128, 9)
    assert model.output_shape[-1] == 6

def test_model_compilation():
    model = build_model((128, 9), 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    assert model.optimizer.get_config()['name'] == 'adam'
    assert model.loss == 'categorical_crossentropy'

def test_model_forward_pass(input_shape, dummy_data):
    model = build_model(input_shape, num_classes=6)
    X, _ = dummy_data
    y_pred = model(X)
    assert y_pred.shape == (10, 6)