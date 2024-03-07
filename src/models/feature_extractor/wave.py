import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
#from tqdm import tqdm
from tqdm.notebook import tqdm
import gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
import warnings

class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        #self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        self.num_rates = len(dilation_rates)
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res
# detail 


from typing import Callable, Optional

import torch
import torch.nn as nn


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class Wavenet_CNNSpectrogram(nn.Module):
    def __init__(
        self,
        in_channels: int = 20,
        base_filters: int | tuple = 128,
        kernel_sizes: tuple = (32, 16, 2),
        stride: int = 2,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        conv: Callable = nn.Conv1d,
        dropout=nn.Dropout1d,
        reinit: bool = True,
    ):
        super().__init__()
        self.wave_block = Wave_Block(in_channels,in_channels, 8, 3)
        self.wave_block2 = Wave_Block(in_channels,in_channels, 2, 3)
        self.out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels,
                    base_filters[0],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=(kernel_sizes[i] - 1) // 2,
                ),
            ]
            
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                #self.spec_conv.append(tmp_block[0])
                self.spec_conv.append(nn.Sequential(*tmp_block))

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        x = self.wave_block(x)

        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            out.append(self.spec_conv[i](x))
        img = torch.stack(out, dim=1)  # (batch_size, out_chans, height, time_steps)
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = img.sigmoid()
        #print(img.shape)
        return img