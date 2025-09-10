# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import torch
from torch import nn


# 一维卷积网络
class convolution1d(nn.Module):
    def __init__(self, embedding_dim=30):
        super(convolution1d, self).__init__()
        # 2*15*1000
        # self.conv2 = nn.Sequential(nn.Conv2d(2, 16, 15))  # 16*1*986

        self.conv1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, 3, 1),  # 32*998  in_channels, out_channels, kernel_size, stride=1, padding=0
            # nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 1),  # 64*996
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),  # 64*497  kernel_size, stride=None, padding=0

            nn.Conv1d(32, 64, 5, 2),  # 64*247
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, 2),  # 128*122
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),  # 128*60

            nn.MaxPool2d(3, 2))  # 31*29

        self.fc = nn.Sequential(
            nn.Linear(31 * 29, 128),  # 1028
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 24))  # 24

    def forward(self, input):
        # feature = self.conv2(input)
        # feature = self.conv1(torch.squeeze(feature, 2))
        # output = self.fc(feature.view(input.shape[0], -1))
        
        # (batch_size, 2, 15, 1000)
        embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 30, 1000)

        # LSTM层与全连接层
        feature = self.conv1(embedding)
        output = self.fc(feature.view(input.shape[0], -1))
        
        return output