import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LiftingLayerMultiD(nn.Module):
    def __init__(self, in_channels):
        super(LiftingLayerMultiD, self).__init__()
        self.to_base = nn.Conv2d(in_channels * 2, in_channels, 1, bias=False)

    def forward(self, x):
        x = torch.cat([F.relu(x, inplace=True), -1.0 * F.relu(-1.0 * x, inplace=True)], dim=1)
        x = self.to_base(x)
        return x


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=True):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.lift1 = LiftingLayerMultiD(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.lift2 = LiftingLayerMultiD(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.lift3 = LiftingLayerMultiD(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                LiftingLayerMultiD(in_planes),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        print(x.size())
        out1 = self.bn1(x)
        out1 = self.lift1(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = self.lift2(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = self.lift3(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.lift1 = LiftingLayerMultiD(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.lift2 = LiftingLayerMultiD(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.lift_out = LiftingLayerMultiD(planes * 4)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.lift1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lift2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.lift_out(out)

        return out


def calc_pad(kernel_size=3, dilation=1):
    kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    return ((kernel_size[0] - 1) * dilation[0] / 2, (kernel_size[1] - 1) * dilation[1] / 2)


class ResNetDepth(nn.Module):
    def __init__(self, num_channels=3, block=Bottleneck, layers=[1, 2, 3, 2], num_elements=199):
        self.inplanes = 64
        super(ResNetDepth, self).__init__()
        self.conv1 = ConvBlock(3, self.inplanes)
        self.bn1 = nn.BatchNorm2d(64)
        self.lift = LiftingLayerMultiD(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        output_size = 29 * 29
        self.fc1 = nn.Linear(512 * block.expansion * output_size, num_elements)
        self.fc2 = nn.Linear(num_elements, num_elements)

        # param initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=3, stride=stride, bias=False, padding=1),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(ConvBlock(self.inplanes, self.inplanes))
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lift(x)
        x = self.maxpool(x)
        #print(x.size())
        #print('layer 1')
        x = self.layer1(x)
        #print(x.size())
        #print('layer 2')
        x = self.layer2(x)
        #print(x.size())
        #print('layer 3')
        x = self.layer3(x)
        #print(x.size())
        #print('layer 4')
        x = self.layer4(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)

        return x

