# Pneumonia Classification

Medical image classification project using PyTorch to classify chest X-rays into three categories: **bacterial pneumonia**, **viral pneumonia**, and **normal**. Uses transfer learning with ResNet18 pretrained on ImageNet.

## Getting Started

### Prerequisites

- Python 3
- PyTorch, TorchVision, NumPy, scikit-learn, matplotlib, seaborn, tqdm
- CUDA-compatible GPU (optional, auto-detected)

### Setup

```bash
source venv/bin/activate
```

### Training

```bash
python train_model.py
```

## Models

Three models were trained and compared in `pneumonia_colab.ipynb`. All use ImageNet-pretrained weights, a custom classification head with Dropout(0.5), CrossEntropyLoss with class weights to handle the dataset imbalance (bacterial pneumonia is overrepresented), and the Adam optimizer.

### Model 1 — ResNet18 (Baseline)

The baseline model with minimal fine-tuning.

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| Architecture       | ResNet18                             |
| Fine-tuning        | Last 10 parameters unfrozen          |
| Augmentation       | Horizontal flip, rotation (±10°), color jitter |
| Scheduler          | StepLR (step_size=5, gamma=0.1)      |
| Epochs             | 10 (fixed)                           |

### Model 2 — ResNet18 (Improved)

Builds on Model 1 with deeper fine-tuning, stronger augmentation, and better training strategies.

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| Architecture       | ResNet18 (configurable)              |
| Fine-tuning        | Entire `layer4` unfrozen             |
| Augmentation       | All of Model 1 + RandomAffine, GaussianBlur, RandomErasing |
| MixUp              | Enabled (alpha=0.2)                  |
| Scheduler          | CosineAnnealingLR                    |
| Epochs             | Up to 30 with early stopping (patience=5) |

### Model 3 — DenseNet121

DenseNet121 is well-suited for medical imaging — Stanford's CheXNet used it for chest X-ray diagnosis. Its dense connections let every layer receive features from all preceding layers, making it parameter-efficient (~8M parameters) while maintaining strong feature extraction.

| Parameter          | Value                                |
|--------------------|--------------------------------------|
| Architecture       | DenseNet121                          |
| Fine-tuning        | `denseblock4` unfrozen               |
| Augmentation       | Same as Model 2                      |
| MixUp              | Enabled (alpha=0.2)                  |
| Scheduler          | CosineAnnealingLR                    |
| Epochs             | Up to 30 with early stopping (patience=5) |

### Model Comparison

| Feature                  | Model 1        | Model 2        | Model 3         |
|--------------------------|----------------|----------------|-----------------|
| Architecture             | ResNet18       | ResNet18       | DenseNet121     |
| Trainable layers         | Last 10 params | layer4 + fc    | denseblock4 + fc|
| Enhanced augmentation    | No             | Yes            | Yes             |
| MixUp                    | No             | Yes            | Yes             |
| LR scheduler             | StepLR         | CosineAnnealing| CosineAnnealing |
| Early stopping           | No             | Yes            | Yes             |

The notebook produces side-by-side confusion matrices and per-class accuracy breakdowns for all three models. Run `pneumonia_colab.ipynb` in Google Colab with a GPU runtime to reproduce the results.

## Potential Improvements

- **Larger dataset** — The current dataset is relatively small (~15,600 training images). A larger and more diverse set of chest X-rays would likely improve generalization, especially for viral pneumonia which has the fewest samples.
- **External datasets** — Combining with other public chest X-ray datasets (e.g., NIH ChestX-ray14, RSNA Pneumonia Detection) could increase volume and variety.
- **More aggressive augmentation** — Techniques like elastic deformation or CLAHE (Contrast Limited Adaptive Histogram Equalization) are commonly used in medical imaging to synthetically expand training data.
- **Larger architectures** — The notebook supports swapping to larger models like ResNet50, EfficientNet-B3, or ConvNeXt-Tiny, which may extract richer features at the cost of more compute.
- **Unfreezing more layers** — Gradually unfreezing earlier layers (progressive fine-tuning) could help the model learn X-ray-specific low-level features rather than relying entirely on ImageNet features.
- **Cross-validation** — Using k-fold cross-validation instead of a single train/test split would give more robust accuracy estimates, especially given the small test set (618 images).

## Data Pipeline

- **Training Augmentation:** Random horizontal flip, rotation (±10°), color jitter (brightness/contrast ±0.2)
- **Normalization:** ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Image Size:** 224×224

## Dataset Structure

```
train/
├── BACTERIAL_PNEUMONIA/  (7,560 images)
├── NORMAL/               (4,044 images)
└── VIRAL_PNEUMONIA/      (4,014 images)
test/
├── BACTERIAL_PNEUMONIA/  (240 images)
├── NORMAL/               (231 images)
└── VIRAL_PNEUMONIA/      (147 images)
```

## Output Artifacts

- `pneumonia_model.pth` — Best model checkpoint (saved when validation accuracy improves)
- `confusion_matrix.png` — Confusion matrix visualization
- `training_history.png` — Loss and accuracy curves

## Key Files

- `train_model.py` — Main training script with complete pipeline (data loading, model creation, training loop, evaluation)
- `pneumonia_colab.ipynb` — Google Colab alternative for cloud-based training
