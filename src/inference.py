import tensorflow as tf
import numpy as np
from pathlib import Path
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


def predict_single_image(model: tf.keras.Model, image_path: str, class_names: list, target_size: tuple = (124, 124)) -> dict:
    """Predict the class of a single image.

    Args:
        model: Trained Keras model.
        image_path: Path to the image file.
        class_names: List of class names.
        target_size: Target size for resizing.

    Returns:
        Dictionary with class probabilities and predicted class.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = tf.keras.utils.load_img(str(path), target_size=target_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) / 255.0

    predictions = model.predict(img_array, verbose=0)[0]

    results = {}
    for i, class_name in enumerate(class_names):
        results[class_name] = float(predictions[i])

    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    return {
        "predictions": results,
        "predicted_class": predicted_class,
        "confidence": confidence,
    }


def predict_images_in_directory(
    model: tf.keras.Model,
    directory_path: str,
    class_names: list,
    target_size: tuple = (124, 124),
) -> list:
    """Predict classes for all images in a directory.

    Args:
        model: Trained Keras model.
        directory_path: Path to directory containing images.
        class_names: List of class names.
        target_size: Target size for resizing.

    Returns:
        List of prediction results for each image.
    """
    dir_path = Path(directory_path)
    if not dir_path.is_dir():
        logger.error(f"Not a valid directory: {directory_path}")
        return []

    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in extensions]

    if not image_files:
        logger.warning(f"No image files found in: {directory_path}")
        return []

    logger.info(f"Found {len(image_files)} images in {directory_path}")
    all_results = []

    for image_file in image_files:
        result = predict_single_image(model, str(image_file), class_names, target_size)
        result["file"] = image_file.name
        all_results.append(result)

        logger.info(f"  {image_file.name}: {result['predicted_class']} ({result['confidence']:.2%})")

    return all_results


def load_and_predict(config: dict) -> list:
    """Load a saved model and predict on images in the evaluation directory.

    Args:
        config: Configuration dictionary.

    Returns:
        List of prediction results.
    """
    model_path = Path(config["paths"]["model_save_path"]) / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"No saved model found at: {model_path}")

    logger.info(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    from src.data_pipeline import create_test_dataset
    _, class_names = create_test_dataset(config)

    eval_dir = config["paths"]["evaluation_data"]
    target_size = tuple(config["data"]["target_size"])

    results = predict_images_in_directory(model, eval_dir, class_names, target_size)

    return results
