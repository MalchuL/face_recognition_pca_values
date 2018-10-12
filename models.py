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


        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn1 = nn.BatchNorm2d(int(out_planes / 2))
        self.selu1 = torch.nn.SELU(inplace=True)

        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 4))
        self.selu2 = torch.nn.SELU(inplace=True)

        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.selu3 = torch.nn.SELU(inplace=True)


        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.SELU(inplace=True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=True),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        #print(x.size())

        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.selu1(out1)


        out2 = self.conv2(out1)
        out2 = self.bn2(out2)
        out2 = self.selu2(out2)


        out3 = self.conv3(out2)
        out3 = self.bn3(out3)
        out3 = self.selu3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.selu1 = nn.SELU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.selu2 = nn.SELU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.selu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.selu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = torch.tanh(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out = torch.cat([ residual,out])

        return out


def calc_pad(kernel_size=3, dilation=1):
    kernel_size = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size
    dilation = (dilation, dilation) if type(dilation) == int else dilation
    return ((kernel_size[0] - 1) * dilation[0] / 2, (kernel_size[1] - 1) * dilation[1] / 2)


class ResNetDepth(nn.Module):
    def __init__(self, num_channels=3, block=Bottleneck, layers=[1, 1, 1, 1], num_elements=199):
        self.inplanes = 32
        super(ResNetDepth, self).__init__()
        self.conv1 = ConvBlock(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.selu1 = nn.SELU(inplace=True)

        self.conv2 = ConvBlock(16, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.selu2 = nn.SELU(inplace=True)

        self.conv3 = ConvBlock(32, self.inplanes)
        self.bn3 = nn.BatchNorm2d(self.inplanes)


        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        output_size = 29 * 29
        self.fc1 = nn.Linear(128 * block.expansion * output_size, 128 * block.expansion // 2, bias=False)
        self.bn4 = nn.BatchNorm2d(128 * block.expansion // 2)
        self.fc2 = nn.Linear(128 * block.expansion // 2, num_elements, bias=False)

        # param initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):

        layers = []
        layers.append(block(self.inplanes, planes, stride, None))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.selu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.selu2(x)

        x = self.conv3(x)
        x = self.bn3(x)

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
        x = self.bn4(x)
        x = torch.tanh(x)
        x = self.fc2(x)


        return x

