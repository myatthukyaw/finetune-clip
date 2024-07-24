# Finetune CLIP

Fine-tune OpenAI's CLIP model for classification tasks on a subset of the ImageNet dataset, ImageNette, and benchmark its performance against state-of-the-art (SOTA) classification models such as ResNets models.


## Installation 

- Install Pytorch - https://pytorch.org/get-started/previous-versions
- Install OpenAI's CLIP - https://github.com/openai/CLIP

## Dataset

Used FastAI's Imagenette dataset for my experiments. You can check it here https://github.com/fastai/imagenette

Prepare the dataset by renaming the directories for each classes to the class names. 
Your Dataset should be like this.

```
_ imagenette2
|__ cassette player
|__ chain_saw
|__ church
|__ english springer 
|__ French_horn
|__ garbage_truck
|__ gas_pump
|__ golf_ball
|__ parachute
|__ tench
```

## Train baseline model

- Available baseline classification models - **resnet18, resnet34,resnet50 , resnet101, resnet152, densenet121, densenet169, densenet201, densenet161, efficientnetb0, googlenet, mobilenet, mobilenetv2, vgg11, vgg13, vgg16, vgg19**.
- Supported Datasets - **ImageNette, Cifar10**


```bash
python train_baseline.py --dataset imagenette --model resnet18 --dataset_path <path to your dataset>
```

## Finetune clip model

- Available CLIP models - **RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px**
- Supported Datasets - **ImageNette**

```bash
python clip_finetune.py --model ViT-B/32 --dataset_path <path to your dataset>
```

## Tensorboard logs

```bash
tensorboard --logdir=runs
```


## Export ONNX

```bash
python scripts/export_onnx.py --input_pytorch_model best.pt --output_onnx_model best.onnx
```

## Quantization

```bash
python scripts/quantize.py --input_pytorch_model best.pt --output_quantized_model best_quantized.pt
```

## References

- (OpenAI CLIP)[https://github.com/openai/CLIP] 
- (Pytorch)[https://pytorch.org/]


## Citations

```
@article{xie2021segformer,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever},
  journal={arXiv:2103.00020v1 [cs.CV] 26 Feb 2021},
  year={2021}
}
```
