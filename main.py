import sys
import tensorflow as tf
from pathlib import Path
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.data_pipeline import set_seed, create_datasets, create_augmentation_layer, create_test_dataset
from src.model import create_cnn_model
from src.training import compile_model, train_model
from src.evaluation import evaluate_model, generate_predictions
from src.inference import predict_images_in_directory
from src.utils.visualization import plot_training_history

logger = setup_logger("melanoma_cnn")


def main():
    """Main pipeline: load config, build model, train, evaluate, predict."""
    try:
        config = load_config("configs/config.yaml")
    except FileNotFoundError:
        logger.error("Config file not found at configs/config.yaml")
        sys.exit(1)

    seed = config["experiment"]["seed"]
    set_seed(seed)

    logger.info("=" * 60)
    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Data version: {config['experiment']['data_version']}")
    logger.info("=" * 60)

    logger.info("Loading datasets...")
    try:
        train_dataset, val_dataset, class_names = create_datasets(config)
    except FileNotFoundError as e:
        logger.error(f"Data directory not found: {e}")
        logger.error("Make sure raw_data/training_set/ exists with benin/ and malin/ subdirectories")
        sys.exit(1)

    num_classes = len(class_names)
    logger.info(f"Classes: {class_names} ({num_classes} total)")

    logger.info("Creating model...")
    model = create_cnn_model(config, num_classes)

    augmentation = create_augmentation_layer(config)
    train_dataset_aug = train_dataset.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    compile_model(model, config)
    model.summary()

    logger.info("Starting training...")
    history = train_model(model, train_dataset_aug, val_dataset, config)

    plots_dir = config["paths"]["plots_dir"]
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    plot_training_history(history, plots_dir)
    logger.info(f"Training plots saved to {plots_dir}")

    logger.info("Evaluating on test set...")
    try:
        test_dataset, _ = create_test_dataset(config)
        metrics = evaluate_model(model, test_dataset, class_names, config)
        generate_predictions(model, test_dataset, class_names, config)
    except FileNotFoundError as e:
        logger.warning(f"Test set not found: {e}")
        logger.warning("Skipping test evaluation")

    logger.info("Running inference on evaluation images...")
    eval_dir = config["paths"]["evaluation_data"]
    target_size = tuple(config["data"]["target_size"])
    if Path(eval_dir).exists():
        results = predict_images_in_directory(model, eval_dir, class_names, target_size)
        for r in results:
            logger.info(f"  {r['file']}: {r['predicted_class']} ({r['confidence']:.2%})")
    else:
        logger.warning(f"Evaluation directory not found: {eval_dir}")

    model_save_path = Path(config["paths"]["model_save_path"]) / "final_model.keras"
    model.save(str(model_save_path))
    logger.info(f"Final model saved to {model_save_path}")

    logger.info("=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
