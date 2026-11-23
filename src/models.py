"""
Model zoo for the CIFAR-10 complexity study.

Five architectures of increasing / different computational character are used:

  MLP          - fully connected baseline, no spatial weight sharing
  SmallCNN     - 2 conv layers, very low FLOPs
  MediumCNN    - VGG-style 6-conv network with BatchNorm
  MobileNetV2  - depthwise-separable inverted residual network (CIFAR variant)
  ResNet18     - deep residual network (CIFAR variant, 11.2M parameters)

All models take a (B, 3, 32, 32) tensor and return (B, 10) logits.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1) Multi-Layer Perceptron
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        dims = [3 * 32 * 32, 1024, 512, 256]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# 2) Small CNN (LeNet-like)
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32 -> 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(inplace=True), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# 3) Medium CNN (VGG-style, 6 conv layers + BN + global average pooling)
# --------------------------------------------------------------------------- #
def _conv_bn(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]


class MediumCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, "M"]
        layers, c = [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += _conv_bn(c, v)
                c = v
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.features(x))
        return self.classifier(torch.flatten(x, 1))


# --------------------------------------------------------------------------- #
# 4) MobileNetV2 (CIFAR variant: first conv stride 1, second stage stride 1)
# --------------------------------------------------------------------------- #
class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, expansion, stride):
        super().__init__()
        self.stride = stride
        hidden = c_in * expansion
        self.conv1 = nn.Conv2d(c_in, hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False)  # depthwise
        self.bn2 = nn.BatchNorm2d(hidden)
        self.conv3 = nn.Conv2d(hidden, c_out, 1, bias=False)  # pointwise projection
        self.bn3 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride == 1 and c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out + self.shortcut(x) if self.stride == 1 else out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        layers, c = [], 32
        for expansion, c_out, n, stride in self.cfg:
            for s in [stride] + [1] * (n - 1):
                layers.append(InvertedResidual(c, c_out, expansion, s))
                c = c_out
        self.layers = nn.Sequential(*layers)
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
# 5) ResNet-18 (CIFAR variant: 3x3 stem, no initial max-pool)
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Sequential()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        layers, c = [], 64
        for c_out, stride in [(64, 1), (128, 2), (256, 2), (512, 2)]:
            for s in [stride, 1]:
                layers.append(BasicBlock(c, c_out, s))
                c = c_out
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


MODEL_REGISTRY = {
    "MLP": MLP,
    "SmallCNN": SmallCNN,
    "MediumCNN": MediumCNN,
    "MobileNetV2": MobileNetV2,
    "ResNet18": ResNet18,
}


def build_model(name: str) -> nn.Module:
    return MODEL_REGISTRY[name]()
