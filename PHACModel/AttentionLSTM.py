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
class lstm_atten_v1(nn.Module):
    def __init__(self, embedding_dim=44, hidden_dim=44, num_layers=3):
        super(lstm_atten_v1, self).__init__()
        # Attention层
        self.attn = SelfAttention(n_layers=2, d_model=embedding_dim, n_heads=4, dropout=0.5)

        # LSTM层
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        # 全连接层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 10, 64),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            # nn.BatchNorm1d(256),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(128, 24)
        )
    
    def forward(self, input):
        # (batch_size, 400, 44)
        # embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 60, 1000)
        embedding = input.transpose(0, 1).contiguous()  # (400, batch_size, 44)

        # mask = get_padding_mask(embedding, 1000)
        # embedding = self.attn(embedding, mask)
        attention = self.attn(embedding)  # (400, batch_size, 44)

        # LSTM层与全连接层
        encoding, _ = self.encoder(attention)  # (400, batch_size, hidden_size)
        encoding = torch.cat((encoding[39], encoding[79], encoding[119], encoding[159], encoding[199], encoding[239], encoding[279], encoding[319], encoding[359], encoding[399]), -1)
        # encoding = encoding[-1]  # 取LSTM最后一个时刻的编码结果 (batch_size, hidden_size)
        decoding = self.decoder(encoding)  # (batch_size, 1)

        # 取每个seq最后一个时刻（即seq的最后一位）的输出作为整个网络的输出
        # decoding = decoding[-1, :, :]  # (batch_size, 24)

        return decoding


# 双向LSTM网络
class lstm_atten_v2(nn.Module):
    def __init__(self, embedding_dim=44, hidden_dim=22, num_layers=3):
        super(lstm_atten_v2, self).__init__()
        # Attention层
        self.attn = SelfAttention(n_layers=2, d_model=embedding_dim, n_heads=4, dropout=0.5)

        # LSTM层
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, bidirectional=True)

        # 全连接层
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 20, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 1),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(128, 24)
        )
    
    def forward(self, input):
        # (batch_size, 400, 44)
        # embedding = torch.cat([input[:, 0, :, :], input[:, 1, :, :]], dim=1)  # (batch_size, 30, 1000)
        # embedding = embedding.transpose(1, 2).contiguous()  # (batch_size, 1000, 30)
        embedding = input.transpose(0, 1).contiguous()  # (400, batch_size, 44)
        seq_len, _, _ = embedding.size()

        attention = self.attn(embedding)  # (400, batch_size, 44)

        # LSTM层
        encoding, _ = self.encoder(attention)  # (400, batch_size, 2*hidden_size)
        # feature = torch.tensor((())).to(DEVICE)
        # for i in range(seq_len):
        #     feature = torch.cat((feature, encoding[i]), dim=-1) 
        # feature = encoding.view(1, batch_size, -1)
        step = int(seq_len / 10)
        feature = torch.cat((encoding[step - 1], encoding[2 * step - 1], encoding[3 * step - 1], encoding[4 * step - 1], encoding[5 * step - 1],
                    encoding[6 * step - 1], encoding[7 * step - 1], encoding[8 * step - 1], encoding[9 * step - 1], encoding[10 * step - 1]), -1)
        # feature = torch.cat((encoding[0], encoding[-1]), -1)  # (batch_size, 4*hidden_size)
        
        # 全连接层
        decoding = self.decoder(feature)  # (batch_size, 1)

        return decoding