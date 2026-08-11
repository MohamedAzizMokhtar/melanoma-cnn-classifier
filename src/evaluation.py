import tensorflow as tf
import numpy as np
from pathlib import Path
from src.utils.logging import setup_logger
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve

logger = setup_logger(__name__)


def evaluate_model(model: tf.keras.Model, test_dataset, class_names: list, config: dict) -> dict:
    """Evaluate model on test dataset and generate reports.

    Args:
        model: Trained Keras model.
        test_dataset: Test dataset.
        class_names: List of class names.
        config: Configuration dictionary.

    Returns:
        Evaluation metrics dictionary.
    """
    logger.info("Evaluating model on test set...")
    results = model.evaluate(test_dataset, verbose=1)

    metrics = {}
    metric_names = model.metrics_names
    for name, value in zip(metric_names, results):
        metrics[name] = float(value)
        logger.info(f"  {name}: {value:.4f}")

    return metrics


def generate_predictions(model: tf.keras.Model, test_dataset, class_names: list, config: dict) -> tuple:
    """Generate predictions and create evaluation plots.

    Args:
        model: Trained Keras model.
        test_dataset: Test dataset.
        class_names: List of class names.
        config: Configuration dictionary.

    Returns:
        Tuple of (true_labels, predicted_labels, predicted_probabilities).
    """
    plots_dir = config["paths"]["plots_dir"]

    y_true = []
    y_pred_proba = []

    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_proba.extend(preds)

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)

    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)

    logger.info("Generating confusion matrix...")
    plot_confusion_matrix(y_true_labels, y_pred_labels, class_names, plots_dir)

    logger.info("Generating ROC curves...")
    plot_roc_curve(y_true_labels, y_pred_proba, class_names, plots_dir)

    correct = np.sum(y_true_labels == y_pred_labels)
    total = len(y_true_labels)
    logger.info(f"Predictions: {correct}/{total} correct ({correct/total*100:.1f}%)")

    return y_true_labels, y_pred_labels, y_pred_proba


def load_and_evaluate(config: dict) -> dict:
    """Load a saved model and evaluate it on the test set.

    Args:
        config: Configuration dictionary.

    Returns:
        Evaluation metrics dictionary.
    """
    model_path = Path(config["paths"]["model_save_path"]) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    from src.data_pipeline import create_test_dataset
    test_dataset, class_names = create_test_dataset(config)

    metrics = evaluate_model(model, test_dataset, class_names, config)
    generate_predictions(model, test_dataset, class_names, config)

    return metrics
