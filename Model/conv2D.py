# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

from torch import nn


# 二维卷积网络
class convolution2d(nn.Module):
    def __init__(self):
        super(convolution2d, self).__init__()
        self.conv = nn.Sequential(
            # 4*120*200
            nn.Conv2d(4, 64, 11, 4),  # 64*28*48  in_channels, out_channels, kernel_size, stride=1, padding=0
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(64, 128, 5, 1, 2),  # 128*28*48
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # 128*13*23  kernel_size, stride=None, padding=0

            nn.Conv2d(128, 256, 3, 1),  # 256*11*21
            # nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1),  # 256*9*19
            # nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1),  # 128*7*17
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # 256*3*8
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 8, 1028),  # 1028
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1028, 512),  # 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 24),  # 24
        )

    def forward(self, input):
        feature = self.conv(input)
        output = self.fc(feature.view(input.shape[0], -1))
        return output