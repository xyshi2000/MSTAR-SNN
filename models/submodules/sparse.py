import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional


class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_value = None
        self.pruning = False
        self.temp = 1

    def _init_mask(self, shape, mean: float, std: float = 0):
        self.mask_value = torch.nn.parameter.Parameter(
            torch.normal(mean, std, size=shape, device='cuda'))
        return self.mask_value

    def _pruning(self, flag: bool):
        self.pruning = flag

    def _set_temp(self, temp: float):
        self.temp = temp

    def mask(self):
        if self.mask_value is None:
            return None
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value.detach())
        else:
            return torch.where(self.mask_value > 0, 1, 0).float()

    def left(self):
        if self.mask_value is None:
            return 0., 0.
        else:
            return self.mask().sum().item(), self.mask_value.numel()

    def forward(self, x):
        if self.mask_value is None:
            return x
        elif self.pruning:
            return torch.sigmoid(self.temp * self.mask_value) * x
        else:
            return torch.where(self.mask_value > 0, x, 0)


class ConvBlock(nn.Module):
    def __init__(self, conv: nn.Conv2d, norm: nn.BatchNorm2d, node: neuron.BaseNode,
                 pool: nn.Module = None, static: bool = False, T: int = None,
                 sparse_neurons: bool = False) -> None:
        super(ConvBlock, self).__init__()
        assert isinstance(conv, nn.Conv2d)
        assert norm is None or isinstance(norm, nn.BatchNorm2d)
        assert node is None or isinstance(node, neuron.BaseNode)
        self.conv = conv
        self.norm = norm
        self.pool = pool
        self.node = node
        self.static = static
        self.T = T
        self.sparse_neurons = sparse_neurons
        self.neuron_mask = Mask()

    def init_mask(self, neurons_mean: float, neurons_std: float = 0):
        masks = []
        if self.sparse_neurons:
            masks.append(
                self.neuron_mask._init_mask(
                    self.node.v.unsqueeze(0).shape, neurons_mean, neurons_std))
        return masks

    def _pruning(self, flag: bool):
        if self.sparse_neurons:
            self.neuron_mask._pruning(flag)

    def set_temp(self, temp: float):
        if self.sparse_neurons:
            self.neuron_mask._set_temp(temp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.conv.weight
        if self.static:
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x.unsqueeze_(0)
            x = x.repeat(self.T, 1, 1, 1, 1)
        else:
            x_shape = [x.shape[0], x.shape[1]]
            x = x.flatten(0, 1)
            x = torch.nn.functional.conv2d(x, weight, bias=self.conv.bias, stride=self.conv.stride,
                                           padding=self.conv.padding, dilation=self.conv.dilation,
                                           groups=self.conv.groups)
            if self.norm is not None:
                x = self.norm(x)
            x_shape.extend(x.shape[1:])
            x = x.view(x_shape)
        if self.pool is not None:
            x = functional.seq_to_ann_forward(x, self.pool)
        if self.node is not None:
            x = self.node(x)
        x = self.neuron_mask(x)
        return x
