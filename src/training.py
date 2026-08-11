import tensorflow as tf
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def get_optimizer(config: dict) -> tf.keras.optimizers.Optimizer:
    """Get optimizer from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Keras optimizer instance.
    """
    training_config = config["training"]
    optimizer_name = training_config["optimizer"].lower()
    learning_rate = training_config["learning_rate"]

    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
        "adamw": tf.keras.optimizers.AdamW,
    }

    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizers[optimizer_name](learning_rate=learning_rate)


def get_callbacks(config: dict) -> list:
    """Create training callbacks from configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        List of Keras callbacks.
    """
    training_config = config["training"]
    paths = config["paths"]
    callbacks = []

    model_save_dir = Path(paths["model_save_path"])
    model_save_dir.mkdir(parents=True, exist_ok=True)

    if training_config.get("model_checkpoint", {}).get("enabled", False):
        ckpt_config = training_config["model_checkpoint"]
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_dir / "best_model.keras"),
            monitor=ckpt_config.get("monitor", "val_accuracy"),
            save_best_only=bool(ckpt_config.get("save_best_only", True)),
            verbose=1,
        ))
        logger.info("ModelCheckpoint callback enabled")

    if training_config.get("early_stopping", {}).get("enabled", False):
        es_config = training_config["early_stopping"]
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=es_config.get("monitor", "val_loss"),
            patience=int(es_config.get("patience", 10)),
            restore_best_weights=True,
            verbose=1,
        ))
        logger.info(f"EarlyStopping enabled (patience={es_config.get('patience', 10)})")

    if training_config.get("lr_scheduler", {}).get("enabled", False):
        lr_config = training_config["lr_scheduler"]
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(lr_config.get("factor", 0.5)),
            patience=int(lr_config.get("patience", 5)),
            min_lr=float(lr_config.get("min_lr", 1e-7)),
            verbose=1,
        ))
        logger.info("ReduceLROnPlateau callback enabled")

    logs_dir = Path(paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    try:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1,
        ))
        logger.info("TensorBoard logging enabled")
    except Exception:
        logger.warning("TensorBoard not available, skipping TensorBoard callback")

    return callbacks


def compile_model(model: tf.keras.Model, config: dict) -> None:
    """Compile model with optimizer, loss, and metrics.

    Args:
        model: Keras model to compile.
        config: Configuration dictionary.
    """
    training_config = config["training"]
    optimizer = get_optimizer(config)

    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(
        optimizer=optimizer,
        loss=training_config["loss"],
        metrics=metrics,
    )

    logger.info(f"Model compiled with {training_config['optimizer']} optimizer")


def train_model(model: tf.keras.Model, train_dataset, val_dataset, config: dict) -> dict:
    """Train the model and return training history.

    Args:
        model: Compiled Keras model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        config: Configuration dictionary.

    Returns:
        Training history dictionary.
    """
    epochs = config["training"]["epochs"]
    callbacks = get_callbacks(config)

    logger.info(f"Starting training for {epochs} epochs")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Training completed")
    return history
