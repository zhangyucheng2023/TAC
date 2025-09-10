# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import torch
from torch import nn

# 输入2*15*1000=30000，输出24个属性值
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 30000, 24, 2048, 512


# 将数据扁平化为一维
class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# 全连接网络
class fullconnection(nn.Module):
    def __init__(self):
        super(fullconnection, self).__init__()
        self.fc = nn.Sequential(
            FlattenLayer(),
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hiddens1, num_hiddens2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hiddens2, num_outputs)
        )

    def forward(self, input):
        return self.fc(input)