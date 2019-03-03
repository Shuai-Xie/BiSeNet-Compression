import torch
import torch.nn as nn
import math

"""
variants of ori resnet
- change fms, output has less fms
- change layers, each layer has less block
- change bottleneck expansion, 4 -> 2
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1  # after basic, fm num not change

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 3x3
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # 3x3
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x  # ori input is residual

        out = self.conv1(x)  # conv1 CBR
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # conv2 CBR, before R, add residual first
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2  # after bottleneck, fm num expands 2 times todo 4->2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  # 1x1 kernel
        self.bn1 = nn.BatchNorm2d(planes)  # BN
        # 3x3
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 and planes expands
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_s(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 16  # 64->16
        super(ResNet_s, self).__init__()
        # 7x7
        self.conv1 = nn.Conv2d(4, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)  # batch norm, each channel normalization = features
        self.relu = nn.ReLU(inplace=True)
        # 3x3
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        :param block: BasicBlock, Bottleneck
        :param planes: plane num
        :param blocks: block num, or a block repeat times
        :param stride: layer1 stride = 1, layer 2,3,4 stride = 2
        :return: a layer of functions
        """
        downsample = None
        # whether to do downsample
        # - inner blocks, no downsample
        # - connect two layers, has downsample and feature expansion
        if stride != 1 or self.inplanes != planes * block.expansion:  # outplanes = planes * block.expansion
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample)
        ]
        self.inplanes = planes * block.expansion  # each time use _make_layer, will update inplanes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.relu(self.bn1(self.conv1(input)))
        x = self.maxpool(x)
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail


def resnet18_s():
    model = ResNet_s(block=BasicBlock, layers=[2, 2, 2, 2])
    return model


def resnet50_s():  # ori [3,4,6,3]
    model = ResNet_s(block=Bottleneck, layers=[2, 3, 3, 2])  # params may reduce
    return model
