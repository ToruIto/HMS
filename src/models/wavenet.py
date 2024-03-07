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
        dilation_rates = [2 ** i for i in range(0,dilation_rates)]
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
class wavenet_classifier(nn.Module):
    def __init__(self, inch=20, kernel_size=3):
        super().__init__()
        self.LSTM = nn.GRU(input_size=128, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)
        #self.wave_block1 = Wave_Block(inch, inch*2, 6, kernel_size)
        #self.wave_block2 = Wave_Block(inch*2, inch*3, 8, kernel_size)
        #self.wave_block3 = Wave_Block(inch*3, inch*4, 4, kernel_size)
        self.wave_block4 = Wave_Block(inch, 128, 6, kernel_size)
        self.fc = nn.Linear(128, 6)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        #x = self.wave_block1(x)
        #x = self.wave_block2(x)
        #x = self.wave_block3(x)

        x = self.wave_block4(x)
        x = x.permute(0, 2, 1)
        x, _ = self.LSTM(x)
        x = torch.mean(x,dim=1)
        x = self.fc(x)
        return x