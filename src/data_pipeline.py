import tensorflow as tf
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    logger.info(f"Random seed set to {seed}")


def create_datasets(config: dict) -> tuple:
    """Create training and validation datasets from directory structure.

    Uses tf.keras.utils.image_dataset_from_directory for modern data loading.
    Applies augmentation as a preprocessing layer during training.

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (train_dataset, val_dataset, class_names).
    """
    data_config = config["data"]
    paths = config["paths"]
    target_size = tuple(data_config["target_size"])
    batch_size = data_config["batch_size"]
    val_split = data_config["validation_split"]
    seed = config["experiment"]["seed"]

    train_dir = Path(paths["train_dir"])
    test_dir = Path(paths["test_dir"])

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    logger.info(f"Loading training data from: {train_dir}")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_size,
        shuffle=True,
        seed=seed,
        validation_split=val_split,
        subset="training",
    )

    logger.info(f"Loading validation data from: {train_dir}")
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
        seed=seed,
        validation_split=val_split,
        subset="validation",
    )

    class_names = train_dataset.class_names
    num_classes = len(class_names)
    logger.info(f"Found {num_classes} classes: {class_names}")

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset, class_names


def create_augmentation_layer(config: dict) -> tf.keras.Sequential:
    """Create a data augmentation preprocessing layer.

    Args:
        config: Configuration dictionary.

    Returns:
        Sequential model serving as augmentation layer.
    """
    aug_config = config["data"]["augmentation"]

    layers = []
    if aug_config.get("horizontal_flip", False):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))
    if aug_config.get("vertical_flip", False):
        layers.append(tf.keras.layers.RandomFlip("vertical"))
    if aug_config.get("rotation_range", 0) > 0:
        layers.append(tf.keras.layers.RandomRotation(aug_config["rotation_range"] / 360))
    if aug_config.get("zoom_range", 0) > 0:
        layers.append(tf.keras.layers.RandomZoom(aug_config["zoom_range"]))
    if aug_config.get("shear_range", 0) > 0:
        layers.append(tf.keras.layers.RandomTranslation(
            aug_config["shear_range"], aug_config["shear_range"]
        ))
    if aug_config.get("brightness_range"):
        low, high = aug_config["brightness_range"]
        layers.append(tf.keras.layers.RandomBrightness((low, high)))

    augmentation = tf.keras.Sequential(layers, name="augmentation")
    logger.info(f"Created augmentation layer with {len(layers)} transforms")
    return augmentation


def create_test_dataset(config: dict) -> tuple:
    """Create a test dataset for evaluation (no augmentation, no shuffle).

    Args:
        config: Configuration dictionary.

    Returns:
        Tuple of (test_dataset, class_names).
    """
    data_config = config["data"]
    paths = config["paths"]
    target_size = tuple(data_config["target_size"])
    batch_size = data_config["batch_size"]

    test_dir = Path(paths["test_dir"])
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
    )

    class_names = test_dataset.class_names
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    return test_dataset, class_names
