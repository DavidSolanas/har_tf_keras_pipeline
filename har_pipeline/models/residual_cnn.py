import tensorflow as tf

def residual_block(x, filters, kernel_size):
    """
    Residual block with two Conv1D layers and skip connection.
    """
    shortcut = x
    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Add()([shortcut, x])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_model(input_shape, num_classes):
    """
    Builds the Residual CNN model for time series classification.

    Args:
        input_shape: tuple, shape of input excluding batch size, e.g. (128, 9)
        num_classes: int, number of classes

    Returns:
        tf.keras.Model object
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = residual_block(x, 64, 3)
    x = residual_block(x, 64, 3)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)
