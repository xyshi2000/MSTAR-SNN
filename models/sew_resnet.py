import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from .submodules.sparse import ConvBlock
from .submodules.layers import conv1x1, conv3x3, create_mslif
from typing import Union, List

__all__ = ['sew_resnet18', 'sew_resnet34', 'sew_resnet50', 'sew_resnet101', 'sew_resnet152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_mslif
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.downsample = downsample
        self.conv1 = ConvBlock(conv3x3(inplanes, planes, stride), norm_layer(planes), activation(),
                               sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(planes, planes), norm_layer(planes), activation(),
                               sparse_neurons=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample: ConvBlock = None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, activation=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = create_mslif
        width = int(planes * (base_width / 64.)) * groups

        self.downsample = downsample
        self.conv1 = ConvBlock(conv1x1(inplanes, width), norm_layer(width), activation(),
                               sparse_neurons=True)
        self.conv2 = ConvBlock(conv3x3(width, width, stride, groups, dilation), norm_layer(width),
                               activation(), sparse_neurons=True)
        self.conv3 = ConvBlock(conv1x1(width, planes * self.expansion),
                               norm_layer(planes * self.expansion), activation(),
                               sparse_neurons=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity

        return out


class SEWResNet(nn.Module):
    def __init__(self, block: Union[BasicBlock, Bottleneck], layers: List[int], num_classes=10,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer: nn.Module = None, T=4, in_channels=1):
        super(SEWResNet, self).__init__()
        self.skip = ['conv1']

        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ConvBlock(
            nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.inplanes), create_mslif(), static=True, T=T,
            sparse_neurons=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))

        if num_classes * 10 < 512 * block.expansion:
            self.fc = nn.Linear(512 * block.expansion, num_classes * 10)
            self.boost = nn.AvgPool1d(10, 10)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)
            self.boost = None

        self.init_weight()
        if zero_init_residual:
            self.zero_init_blocks()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def zero_init_blocks(self):
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.conv3.norm.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.conv2.norm.weight, 0)

    def _make_layer(self, block: Union[BasicBlock, Bottleneck], planes: int, blocks: int, stride=1,
                    dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ConvBlock(conv1x1(self.inplanes, planes * block.expansion, stride),
                                   norm_layer(planes * block.expansion), create_mslif(),
                                   sparse_neurons=True)

        layers: List[Union[BasicBlock, Bottleneck]] = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width,
                  previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                      dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        out = functional.seq_to_ann_forward(x, self.fc)

        if self.boost is not None:
            #### [T, N, L] -> [T, N, C=1, L]
            out = out.unsqueeze(2)
            out = functional.seq_to_ann_forward(out, self.boost).squeeze(2)

        return out

    def forward(self, x: torch.Tensor):
        return self._forward_impl(x)


def _sew_resnet(block, layers, **kwargs):
    model = SEWResNet(block, layers, **kwargs)
    return model


def sew_resnet18(**kwargs):
    return _sew_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(**kwargs):
    return _sew_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(**kwargs):
    return _sew_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(**kwargs):
    return _sew_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)
