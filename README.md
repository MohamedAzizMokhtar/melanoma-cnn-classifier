# Melanoma CNN Classifier

A deep learning pipeline for binary classification of melanoma skin lesions (benign vs malignant) using Convolutional Neural Networks with MLOps best practices.

## Project Structure

```
.
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── data_pipeline.py         # Data loading, augmentation, preprocessing
│   ├── model.py                 # CNN architecture (custom + transfer learning)
│   ├── training.py              # Training loop with callbacks
│   ├── evaluation.py            # Model evaluation and metrics
│   ├── inference.py             # Prediction on new images
│   ├── versioning.py            # DVC data versioning
│   └── utils/
│       ├── config.py            # Configuration loader
│       ├── logging.py           # Logging setup
│       └── visualization.py     # Plot generation (accuracy, confusion matrix, ROC)
├── raw_data/                    # Original dataset (training/test splits)
│   ├── training_set/
│   │   ├── benin/
│   │   └── malin/
│   └── test_set/
│       ├── benin/
│       └── malin/
├── evaluation_data/             # Unseen images for manual evaluation
├── outputs/                     # Generated artifacts
│   ├── models/                  # Saved model checkpoints
│   ├── plots/                   # Training curves, confusion matrix, ROC
│   └── logs/                    # TensorBoard logs
├── main.py                      # Main pipeline entry point
├── dvc.yaml                     # DVC pipeline definition
├── requirements.txt             # Python dependencies
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.9+
- TensorFlow 2.10+
- DVC (optional, for data versioning)

### Installation

```bash
# Clone the repository
git clone https://github.com/MohamedAzizMokhtar/melanoma-cnn-classifier.git
cd melanoma-cnn-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place your melanoma dataset in `raw_data/` with the following structure:

```
raw_data/
├── training_set/
│   ├── benin/       # Benign lesion images
│   └── malin/       # Malignant lesion images
├── test_set/
│   ├── benin/
│   └── malin/
└── cas.jpg          # Sample image for single prediction
```

## Usage

### Run Full Pipeline

```bash
python main.py
```

This will:
1. Load and preprocess the dataset
2. Apply data augmentation
3. Train the CNN model with callbacks (early stopping, LR scheduling, checkpointing)
4. Evaluate on the test set
5. Generate plots (training curves, confusion matrix, ROC curves)
6. Run inference on evaluation images
7. Save the trained model

### Configuration

All settings are in `configs/config.yaml`:

- **experiment**: Name, data version, random seed
- **paths**: All directory paths (no hardcoded paths)
- **data**: Image size, batch size, augmentation parameters
- **training**: Epochs, learning rate, optimizer, callbacks
- **model**: Architecture choice, dropout rate, transfer learning options

### Data Versioning (DVC)

```bash
# Track new data version
python -m src.versioning

# Pull a specific data version
dvc pull
```

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

## Model Architecture

### Custom CNN
- 4 convolutional blocks with BatchNormalization
- Global Average Pooling
- Dense layers with Dropout regularization
- Softmax output for binary classification

### Transfer Learning (optional)
Enable in `configs/config.yaml`:
```yaml
model:
  use_transfer_learning: true
  transfer_learning_model: "EfficientNetB0"  # or ResNet50, VGG16
```

## Outputs

After running the pipeline, check `outputs/`:

- `outputs/models/best_model.keras` - Best model checkpoint
- `outputs/models/final_model.keras` - Final trained model
- `outputs/plots/training_curves.png` - Accuracy and loss plots
- `outputs/plots/confusion_matrix.png` - Test set confusion matrix
- `outputs/plots/roc_curves.png` - Per-class ROC curves
- `outputs/logs/` - TensorBoard event files

## License

MIT
