import tensorflow as tf
from tensorflow.keras import layers, models
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def create_cnn_model(config: dict, num_classes: int) -> tf.keras.Model:
    """Create a CNN model for melanoma classification.

    Supports both a custom CNN architecture and transfer learning.

    Args:
        config: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    model_config = config["model"]
    input_shape = tuple(model_config["input_shape"])
    dropout_rate = model_config.get("dropout_rate", 0.3)
    dense_units = model_config.get("dense_units", 128)

    if model_config.get("use_transfer_learning", False):
        model = _create_transfer_model(config, num_classes)
    else:
        model = _create_custom_cnn(input_shape, num_classes, dropout_rate, dense_units)

    logger.info(f"Model created: {model.count_params()} parameters")
    return model


def _create_custom_cnn(
    input_shape: tuple,
    num_classes: int,
    dropout_rate: float,
    dense_units: int,
) -> tf.keras.Model:
    """Build a custom CNN architecture.

    Args:
        input_shape: Input image shape.
        num_classes: Number of output classes.
        dropout_rate: Dropout rate for regularization.
        dense_units: Number of units in dense layer.

    Returns:
        Compiled Keras model.
    """
    model = models.Sequential(name="custom_cnn")

    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(dense_units))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(dense_units // 2, activation="relu"))
    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(num_classes, activation="softmax"))

    return model


def _create_transfer_model(config: dict, num_classes: int) -> tf.keras.Model:
    """Create a model using transfer learning.

    Args:
        config: Configuration dictionary.
        num_classes: Number of output classes.

    Returns:
        Compiled Keras model.
    """
    model_config = config["model"]
    input_shape = tuple(model_config["input_shape"])
    base_model_name = model_config.get("transfer_learning_model", "EfficientNetB0")
    dropout_rate = model_config.get("dropout_rate", 0.3)

    base_models = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
        "VGG16": tf.keras.applications.VGG16,
    }

    if base_model_name not in base_models:
        raise ValueError(f"Unsupported transfer learning model: {base_model_name}")

    logger.info(f"Loading pretrained {base_model_name} as base model")
    base_model = base_models[base_model_name](
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )
    base_model.trainable = False

    model = models.Sequential(name=f"transfer_{base_model_name}")
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(num_classes, activation="softmax"))

    return model
