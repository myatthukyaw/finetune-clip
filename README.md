# Fineune CLIP
## Overview

This repo contains training pipeline for deep learning models using PyTorch. It supports training on CIFAR-10 and custom datasets and integrates features like TensorBoard logging, custom transformations, and flexible configurations for training parameters.
Also contain finetuning OpenAI's CLIP model for classification tasks on a subset of the ImageNet dataset, ImageNette

## Motivation

CLIP is being used for various tasks ranging from semantic image search to zero-shot image labeling. It also plays a crucial role in the architecture of Stable Diffusion and is integral to the recently emerging field of large multimodal models (LMMs). This repo will use CLIP for classification tasks and benchmark its performace against the baseline classification models. 

## Features
- **Dataset Support**: CIFAR-10 and custom datasets.
- **Model Selection**: Pytorch native classificaton models, Custom classification models and OpenAI's CLIP
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
- **`models/`**: Contains model definitions. The `get_model` function loads the selected architecture.
- **`scripts/`**: Includes utilities for exporting models to ONNX format and performing model quantization.
- **`tools/`**: Contains the core functionalities for training, evaluation, and metrics.

## Usage

### Command-Line Arguments
Run the script with the following arguments:

| Argument              | Default                     | Description                                                                 |
|-----------------------|-----------------------------|-----------------------------------------------------------------------------|
| `--model`             | `resnet18`                 | Model architecture to use.                                                 |
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

### Example Usage
#### CIFAR-10 Dataset
```bash
python train_baseline.py --dataset cifar10 --model resnet18 --batch_size 32 --epochs 50 --tensorboard --lr_scheduler --gradient_clipping
```

#### Custom Dataset
```bash
python train_baseline.py --dataset custom --dataset_path /path/to/custom/dataset --model resnet50 --tensorboard --lr_scheduler --gradient_clipping
```

#### CLIP Training with Custom Dataset

## Finetune clip model

- Available CLIP models - **RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px**

```bash
python train_clip.py --dataset custom --dataset_path /path/to/custom/dataset --model ViT-B/32 --tensorboard --lr_scheduler --gradient_clipping
```

## Dataset Preparation

### CIFAR-10
The CIFAR-10 dataset is downloaded automatically if selected.

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

Used FastAI's Imagenette dataset for my experiments. You can check it here https://github.com/fastai/imagenette
Prepare the dataset by renaming the directories for each classes to the class names. 

```
_ imagenette2/
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
```

## Output
Model checkpoints and logs are saved in the `exps/` directory. TensorBoard logs are stored under the `runs/` folder.

## Export ONNX

```bash
python scripts/export_onnx.py --input_pytorch_model best.pt --output_onnx_model best.onnx
```
This only works for models like ResNet, DenseNet models. Still working for CLIP model. 

## Quantization

```bash
python scripts/quantize.py --input_pytorch_model best.pt --output_quantized_model best_quantized.pt
```

## Acknowledgments
- [OpenAI CLIP](https://github.com/openai/CLIP) 
- [Pytorch](https://pytorch.org/)
