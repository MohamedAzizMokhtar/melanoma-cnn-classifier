import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc


def plot_training_history(history, output_dir: str = "outputs/plots") -> None:
    """Plot and save training accuracy and loss curves.

    Args:
        history: Keras training history object.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Model Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Model Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: str = "outputs/plots") -> None:
    """Plot and save a confusion matrix.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of class names.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_pred_proba, class_names, output_dir: str = "outputs/plots") -> None:
    """Plot and save ROC curves for each class.

    Args:
        y_true: True labels (one-hot encoded).
        y_pred_proba: Predicted probabilities.
        class_names: List of class names.
        output_dir: Directory to save plots.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true_binary = np.eye(n_classes)[y_true] if y_true.ndim == 1 else y_true

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab10", n_classes)

    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], color=colors(i), linewidth=2,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
