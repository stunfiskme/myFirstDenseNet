import torch
import torch.nn as nn
import torch.nn.functional as F


# DenseNet BottleNeck layer
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Transition Layer
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


# DenseNet Model
class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 24, 16), growth_rate=12, reduction=0.5, num_classes=106):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        num_init_features = 2 * growth_rate

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)

        # Dense blocks and transition layers
        self.dense1 = DenseBlock(num_init_features, block_config[0], growth_rate)
        num_features = num_init_features + block_config[0] * growth_rate
        out_features = int(num_features * reduction)
        self.trans1 = Transition(num_features, out_features)

        self.dense2 = DenseBlock(out_features, block_config[1], growth_rate)
        num_features = out_features + block_config[1] * growth_rate
        out_features = int(num_features * reduction)
        self.trans2 = Transition(num_features, out_features)

        self.dense3 = DenseBlock(out_features, block_config[2], growth_rate)
        num_features = out_features + block_config[2] * growth_rate
        out_features = int(num_features * reduction)
        self.trans3 = Transition(num_features, out_features)

        self.dense4 = DenseBlock(out_features, block_config[3], growth_rate)
        num_features = out_features + block_config[3] * growth_rate

        # Global average pooling and fully connected layer
        self.bn = nn.BatchNorm2d(num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = self.pool(F.relu(self.bn(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

