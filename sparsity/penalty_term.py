import torch
from torch import Tensor, nn
from models.submodules.sparse import ConvBlock
from typing import List


class PenaltyTerm(nn.Module):
    """Apply the same penalty coefficient to every prunable neuron."""

    def __init__(self, model: nn.Module, lmbda: float) -> None:
        super(PenaltyTerm, self).__init__()
        self.layers: List[ConvBlock] = []
        self.model = model
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.layers.append(m)
        self.lmbda = lmbda

    def forward(self) -> Tensor:
        loss = 0
        for layer in self.layers:
            if layer.sparse_neurons:
                loss = loss + self.lmbda * torch.sigmoid(
                    layer.neuron_mask.mask_value * layer.neuron_mask.temp).sum()
        return loss
