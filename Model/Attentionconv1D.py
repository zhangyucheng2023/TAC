# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import copy
import torch
from torch import nn
from utils import get_padding_mask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super(SelfAttentionLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        src2 = self.multihead_attn(src, src, src, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        return src



class SelfAttention(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, dropout=0.0):
        super(SelfAttention, self).__init__()
        self.n_layers = n_layers
        self.layers = _get_clones(SelfAttentionLayer(d_model, n_heads, dropout), n_layers)

    def forward(self, x, x_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=x_padding_mask)
        return x


# 单向LSTM网络
class convolution1d_atten(nn.Module):
    def __init__(self, embedding_dim=30, atten_layers=2):
        super(convolution1d_atten, self).__init__()
        # 2*15*1000
        # Attention层
        self.attn = SelfAttention(n_layers=atten_layers, d_model=embedding_dim, n_heads=3, dropout=0.5)  # 30*1000

        self.conv1 = nn.Sequential(
            nn.Conv1d(embedding_dim, 32, 3, 1),  # 32*998  in_channels, out_channels, kernel_size, stride=1, padding=0
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(32, 32, 3, 1),  # 32*996
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),  # 32*497  kernel_size, stride=None, padding=0

            nn.Conv1d(32, 64, 5, 2),  # 64*247
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(64, 64, 5, 2),  # 64*122
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(3, 2),  # 64*60

            nn.MaxPool2d(3, 2))  # 31*29

        self.fc = nn.Sequential(
            nn.Linear(31 * 29, 128),  # 128
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 24))  # 24
    
    def forward(self, input):
        # (batch_size, 2, 15, 1000)
        embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 30, 1000)
        embedding = embedding.transpose(1, 2).contiguous()  # (batch_size, 1000, 30)
        embedding = embedding.transpose(0, 1).contiguous()  # (1000, batch_size, 30)

        # mask = get_padding_mask(embedding, 1000)
        # embedding = self.attn(embedding, mask)
        attention = self.attn(embedding)  # (1000, batch_size, 30)
        attention = attention.transpose(0, 1).contiguous()  # (batch_size, 1000, 30)
        attention = attention.transpose(1, 2).contiguous()  # (batch_size, 30, 1000)

        # LSTM层与全连接层
        feature = self.conv1(attention)
        output = self.fc(feature.view(input.shape[0], -1))

        return output




# self.conv1 = nn.Sequential(
#             nn.Conv1d(30, 32, 3, 1),  # 64*998  in_channels, out_channels, kernel_size, stride=1, padding=0
#             # nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Conv1d(32, 64, 3, 1),  # 64*996
#             # nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.MaxPool1d(3, 2),  # 64*497  kernel_size, stride=None, padding=0

#             nn.Conv1d(64, 128, 5, 2),  # 128*247
#             # nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 256, 5, 2),  # 256*122
#             # nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.MaxPool1d(3, 2),  # 256*60
            
#             nn.MaxPool2d(3, 2))  # 127*29

#         self.fc = nn.Sequential(
#             nn.Linear(127 * 29, 1028),  # 1028
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(1028, 512),  # 512
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 24),  # 24
#         )