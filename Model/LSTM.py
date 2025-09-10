# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import torch
from torch import nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 单向LSTM网络
class lstm_v1(nn.Module):
    def __init__(self, embedding_dim=30, hidden_dim=30, lstm_layers=3):
        super(lstm_v1, self).__init__()
        # LSTM层
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, lstm_layers)

        # 全连接层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 10, 128),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 24),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(128, 24)
        )
    
    def forward(self, input):
        # (batch_size, 2, 15, 1000)
        embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 30, 1000)
        embedding = embedding.transpose(1, 2).contiguous()  # (batch_size, 1000, 30)
        embedding = embedding.transpose(0, 1).contiguous()  # (1000, batch_size, 30)
        seq_len, _, _ = embedding.size()

        # LSTM层与全连接层
        encoding, _ = self.encoder(embedding)  # (1000, batch_size, hidden_size)
        step = int(seq_len / 10)
        feature = torch.cat((encoding[step - 1], encoding[2 * step - 1], encoding[3 * step - 1], encoding[4 * step - 1], encoding[5 * step - 1],
                    encoding[6 * step - 1], encoding[7 * step - 1], encoding[8 * step - 1], encoding[9 * step - 1], encoding[10 * step - 1]), -1)
        # encoding = encoding[-1]  # 取LSTM最后一个时刻的编码结果 (batch_size, hidden_size)
        decoding = self.decoder(feature)  # (batch_size, 24)

        # 取每个seq最后一个时刻（即seq的最后一位）的输出作为整个网络的输出
        # decoding = decoding[-1, :, :]  # (batch_size, 24)

        return decoding


# 双向LSTM网络
class lstm_v2(nn.Module):
    def __init__(self, embedding_dim=30, hidden_dim=15, lstm_layers=3):
        super(lstm_v2, self).__init__()
        # LSTM层
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, lstm_layers, bidirectional=True)

        # 全连接层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 20, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 24),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(128, 24)
        )
    
    def forward(self, input):
        # (batch_size, 2, 15, 1000)
        embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 30, 1000)
        embedding = embedding.transpose(1, 2).contiguous()  # (batch_size, 1000, 30)
        embedding = embedding.transpose(0, 1).contiguous()  # (1000, batch_size, 30)
        seq_len, _, _ = embedding.size()

        # LSTM层
        encoding, _ = self.encoder(embedding)  # (1000, batch_size, 2*hidden_size)
        # feature = torch.tensor((())).to(DEVICE)
        # for i in range(seq_len):
        #     feature = torch.cat((feature, encoding[i]), dim=-1) 
        # feature = encoding.view(1, batch_size, -1)
        # 每隔100帧的隐含层输入到全连接层
        step = int(seq_len / 10)
        feature = torch.cat((encoding[step - 1], encoding[2 * step - 1], encoding[3 * step - 1], encoding[4 * step - 1], encoding[5 * step - 1],
                    encoding[6 * step - 1], encoding[7 * step - 1], encoding[8 * step - 1], encoding[9 * step - 1], encoding[10 * step - 1]), -1)
        
        # 全连接层
        decoding = self.decoder(feature)  # (batch_size, 24)

        return decoding