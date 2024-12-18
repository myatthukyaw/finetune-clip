from .densenet import (DenseNet121, DenseNet161, DenseNet169, DenseNet201,
                       densenet_cifar)
from .efficientnet import EfficientNetB0
from .googlenet import GoogLeNet
from .mobilenet import MobileNet
from .mobilenetv2 import MobileNetV2
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG


def get_model(model_name, num_classes=10):
    models = {
        'densenet121' : DenseNet121(num_classes=num_classes),
        'densenet169' : DenseNet169(num_classes=num_classes),
        'densenet201' : DenseNet201(num_classes=num_classes),
        'densenet161' : DenseNet161(num_classes=num_classes),
        'densenetcifar' : densenet_cifar(num_classes=num_classes),
        'efficientnetb0' : EfficientNetB0(),
        'googlenet' : GoogLeNet(),
        'mobilenet' : MobileNet(),
        'mobilenetv2' : MobileNetV2(),
        'resnet18' : ResNet18(num_classes=num_classes),
        'resnet34' : ResNet34(num_classes=num_classes),
        'resnet50' : ResNet50(num_classes=num_classes),
        'resnet101' : ResNet101(num_classes=num_classes),
        'resnet152' : ResNet152(num_classes=num_classes),
        'vgg11' : VGG('VGG11', num_classes=num_classes),
        'vgg13' : VGG('VGG13', num_classes=num_classes),
        'vgg16' : VGG('VGG16', num_classes=num_classes),
        'vgg19' : VGG('VGG19', num_classes=num_classes),
    }
    model = models.get(model_name.lower())  # Handle case insensitivity
    if model is None:
        raise ValueError("Model name not recognized")
    return model