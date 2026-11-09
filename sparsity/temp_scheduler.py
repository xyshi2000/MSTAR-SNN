import math
from torch import nn
from models.submodules.sparse import ConvBlock
from typing import List


class TemperatureScheduler:
    def __init__(self, model: nn.Module, init_temp: float, final_temp: float, T0: int,
                 Tmax: int) -> None:
        assert init_temp > 0
        assert init_temp <= final_temp
        assert T0 < Tmax
        self.conv_blocks: List[ConvBlock] = []
        for m in model.modules():
            if isinstance(m, ConvBlock):
                self.conv_blocks.append(m)
                m.set_temp(init_temp)
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.T0 = T0
        self.Tmax = Tmax
        self.current_step = 0
        self.factor = math.log(self.final_temp / self.init_temp)

    def _temp(self):
        if self.current_step < self.T0:
            return self.init_temp
        if self.current_step > self.Tmax:
            return self.final_temp
        return self.init_temp * math.exp(self.factor * (self.current_step - self.T0) /
                                         (self.Tmax - self.T0))

    def step(self):
        self.current_step = self.current_step + 1
        temp = self._temp()
        for block in self.conv_blocks:
            block.set_temp(temp)
        return

    def __str__(self) -> str:
        return 'temperature: {:.3e}'.format(self._temp())
