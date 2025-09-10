# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import torch


# 将网络的输出结果转化为对形容词的预测，方便计算准确率等指标
def TensorValue2d_to_PredictLabel(output):
    m = output.shape[0]
    n = output.shape[1]
    for i in range(m):
        for j in range(n):
            if output[i, j] <= 0.5:  # 小于0.5的结果置0，认为没有该形容词取值
                output[i, j] = 0
            else:  # 大于0.5的结果置1，认为具有该形容词取值
                output[i, j] = 1
    return output


# 计算当前batch每条数据形容词预测准确率之和（之所以求和不求均值是因为样本总数太少，最后一个batch的样本影响可能会很大）
def Accuracy_Sum(output, label):
    batch = output.shape[0]
    adj = output.shape[1]
    cnt = 0
    for i in range(batch):
        for j in range(adj):
            if output[i][j] == label[i][j]:  # 形容词预测正确
                cnt = cnt + 1
    return cnt / adj


# 计算当前batch每条数据形容词预测混淆矩阵
def F1_Score_Sum(output, label):
    batch = output.shape[0]
    adj = output.shape[1]
    TP = FP = TN = FN = 0
    for i in range(batch):
        for j in range(adj):
            if output[i][j] == 1 and label[i][j] == 1:
                TP += 1
            if output[i][j] == 1 and label[i][j] == 0:
                FP += 1
            if output[i][j] == 0 and label[i][j] == 0:
                TN += 1
            if output[i][j] == 0 and label[i][j] == 1:
                FN += 1
    return TP, FP, TN, FN


def get_padding_mask(x, x_lens):
    """
    :param x: (seq_len, batch_size, feature_dim)
    :param x_lens: sequence lengths within a batch with size (batch_size,)
    :return: padding_mask with size (batch_size, seq_len)
    """
    seq_len, batch_size, _ = x.size()
    mask = torch.ones(batch_size, seq_len, device=x.device)

    for seq, seq_len in enumerate(x_lens):
        mask[seq, :seq_len] = 0
    mask = mask.bool()
    return mask