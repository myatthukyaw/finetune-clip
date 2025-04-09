# Finetune CLIP
## Overview

This repo contains a training pipeline for fine-tuning OpenAI's CLIP model using PyTorch. It supports training on CIFAR-10 and custom datasets and integrates features like TensorBoard logging, custom transformations, and flexible configurations for training parameters.

## Motivation

CLIP is being used for various tasks ranging from semantic image search to zero-shot image labeling. It also plays a crucial role in the architecture of Stable Diffusion and is integral to the recently emerging field of large multimodal models (LMMs). This repo fine-tunes CLIP for classification tasks and benchmarks its performance against baseline classification models.

## Features
- **Dataset Support**: CIFAR-10 and custom datasets.
- **Model Selection**: 
  - CLIP models (ViT and ResNet variants)
  - Baseline models (ResNet, DenseNet, etc.) for comparison
- **Training Features**:
  - Label smoothing
  - Learning rate scheduling
  - Learning rate warmup strategy
  - Gradient clipping
  - Optimizer choice (SGD or Adam)
- **Logging**: TensorBoard integration for monitoring training metrics.

## Installation
The script requires the following Python packages:
- Python >= 3.8
- torchvision
- TensorBoard
- Install Pytorch - https://pytorch.org/get-started/previous-versions
- Install OpenAI's CLIP - https://github.com/openai/CLIP

You can install the dependencies using pip:
```bash
pip install torch torchvision tensorboard
pip install git+https://github.com/openai/CLIP.git
```

## File Structure
- **`src/train_clip.py`**: Main script for fine-tuning CLIP models.
- **`src/train_baseline.py`**: Script for training conventional classification models.
- **`src/inference.py`**: Script for running inference with trained models.
- **`src/trainer/`**: Contains trainer classes for different model types:
  - `clip_trainer.py`: CLIP-specific training implementation.
  - `trainer.py`: Baseline trainer for standard classification models.
  - `base.py`: Base trainer class with common functionality.
- **`src/tools/`**: Contains utilities for datasets, metrics, and other helper functions.

## Trainers
The repo includes two main training classes:

### Baseline Trainer
The standard classification trainer (`src/trainer/trainer.py`) provides:
- Standard image classification training pipeline
- Single-label classification using cross-entropy or binary cross-entropy loss
- Standard metrics evaluation and logging
- Used with torchvision models like ResNet, DenseNet, etc.

### CLIP Trainer
The CLIP-specific trainer (`src/trainer/clip_trainer.py`) provides:
- Image-text contrastive learning
- Custom CIFAR10 dataset adaptation through the CIFAR10Wrapper class
- Specialized training for dual-encoder architecture
- Support for CLIP's specific model architecture and inference pattern

## Usage

### Command-Line Arguments for Training
Run the training scripts with the following arguments:

| Argument              | Default                     | Description                                                                 |
|-----------------------|-----------------------------|-----------------------------------------------------------------------------|
| `--model`             | `ViT-B/32`                 | CLIP model architecture to use.                                            |
| `--device`            | `cuda`                     | Device to use for training (`cuda` or `cpu`).                               |
| `--epochs`            | `30`                       | Number of training epochs.                                                 |
| `--batch_size`        | `8`                        | Batch size for training and validation.                                     |
| `--optimizer`         | `SGD`                      | Optimizer choice (`SGD` or `Adam`).                                        |
| `--loss`              | `CE`                       | Loss function (`CE` for Cross Entropy or `BCE` for Binary Cross Entropy).  |
| `--label_smoothing`   | `0.1`                      | Apply label smoothing to reduce overconfidence.                            |
| `--lr`                | `0.01`                     | Initial learning rate.                                                     |
| `--momentum`          | `0.937`                    | Momentum value for SGD optimizer.                                          |
| `--weight_decay`      | `0.0005`                   | Weight decay for regularization.                                           |
| `--dataset`           | `custom`                   | Dataset to use (`cifar10` or `custom`).                                    |
| `--dataset_path`      | `datasets/imagenette2`     | Path to the custom dataset.                             |
| `--lr_scheduler`      | `False`                    | Use a learning rate scheduler.                                             |
| `--warmup`            | `False`                    | Use a warmup strategy for training.                                        |
| `--gradient_clipping` | `False`                    | Enable gradient clipping to stabilize training.                            |
| `--tensorboard`       | `False`                    | Log training metrics to TensorBoard.                                       |

### Command-Line Arguments for Inference
The inference script (`src/inference.py`) offers the following options:

| Argument              | Default                     | Description                                                                 |
|-----------------------|-----------------------------|-----------------------------------------------------------------------------|
| `--model_path`        | (required)                 | Path to the fine-tuned CLIP model.                                          |
| `--image_path`        | (required)                 | Path to a single image or directory of images for inference.                |
| `--clip_model`        | `ViT-B/32`                 | CLIP model architecture to use.                                            |
| `--top_k`             | `3`                        | Number of top predictions to show.                                         |
| `--classes`           | (auto-detect)              | Classes to use for inference (if not provided, inferred from directories). |
| `--output_dir`        | `inference-results`        | Directory to save visualization results.                                    |
| `--batch_size`        | `8`                        | Batch size for inference with multiple images.                             |
| `--device`            | `cuda`                     | Device to run inference on (`cuda` or `cpu`).                              |
| `--visualization`     | `False`                    | Enable visualization of results on images.                                  |
| `--threshold`         | `0.5`                      | Confidence threshold for predictions.                                      |

### Example Usage
#### CIFAR-10 Dataset with CLIP
```bash
python -m src.train_clip --dataset cifar10 --model ViT-B/32 --batch_size 32 --epochs 20 --tensorboard --lr_scheduler
```

#### Custom Dataset with CLIP
```bash
python -m src.train_clip --dataset custom --dataset_path /path/to/custom/dataset --model ViT-B/16 --tensorboard --lr_scheduler --gradient_clipping
```

#### Baseline Model Training
```bash
python -m src.train_baseline --dataset cifar10 --model resnet18 --batch_size 32 --epochs 50 --tensorboard --lr_scheduler
```

#### Running Inference
```bash
# Basic usage with a single image
python -m src.inference --model_path exps/clip-finetuned/ViT-B-32_best_model.pth --image_path path/to/image.jpg

# Process a directory of images with visualization
python -m src.inference --model_path exps/clip-finetuned/ViT-B-32_best_model.pth --image_path path/to/image_dir --visualization

# Specify custom classes
python -m src.inference --model_path exps/clip-finetuned/ViT-B-32_best_model.pth --image_path path/to/image_dir --classes dog cat bird
```

## Dataset Handling

### CIFAR-10
The CIFAR-10 dataset is downloaded automatically if selected. A specialized `CIFAR10Wrapper` class adapts the dataset to the format expected by the CLIP trainer.

### Custom Dataset
The custom dataset should be organized as follows:
```
/dataset_path/
    train/
        class1/
            image1.jpg
            image2.jpg
            ...
        class2/
            ...
    val/
        class1/
            ...
        class2/
            ...
```

### Imagenette Dataset

The repo was tested with FastAI's Imagenette dataset: https://github.com/fastai/imagenette
Prepare the dataset by organizing it as follows:

```
_ imagenette2/
 |__ train/
     |__ cassette player/
     |__ chain_saw/
     |__ church/
     |__ english springer/
     |__ French_horn/
     |__ garbage_truck/
     |__ gas_pump/
     |__ golf_ball/
     |__ parachute/
     |__ tench/
 |__ val/
     |__ cassette player/
     |__ chain_saw/
     |__ ...
```

## Output and Results

### Training Output
Model checkpoints and logs are saved in the `exps/` directory. TensorBoard logs are stored under the `runs/` folder. The training process displays comprehensive metrics including:

- Class-wise accuracy tables
- Detailed evaluation metrics for precision, recall, and F1-score
- Progress bars with loss and accuracy metrics
- Confusion matrices saved to the experiment directory

### Inference Output
Inference results are displayed in a well-formatted table and optionally saved as annotated images in the `inference-results` directory. Results include:

- Predictions with confidence scores
- Overall accuracy when ground truth is available
- Color-coded confidence indicators
- Batch processing for efficient evaluation of multiple images

## Available Models

### CLIP Models
The following CLIP models are available for fine-tuning:
- **ResNet variants**: RN50, RN101, RN50x4, RN50x16, RN50x64
- **ViT variants**: ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

### Baseline Models
Standard classification models from torchvision:
- **ResNet variants**: resnet18, resnet34, resnet50, resnet101
- **DenseNet variants**: densenet121, densenet169, densenet201
- **Other architectures**: mobilenet_v2, efficientnet_b0, etc.

## Acknowledgments
- [OpenAI CLIP](https://github.com/openai/CLIP) 
- [Pytorch](https://pytorch.org/)
- [FastAI Imagenette](https://github.com/fastai/imagenette)
