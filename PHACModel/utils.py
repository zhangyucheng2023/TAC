# -*- coding: utf-8 -*-
# author:Yuanpei Zhang

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PHACreader import File_handler


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


# 数据集类，获取数据和标签，从torch的DataSet类继承
class TrailDataSet(Dataset):
    def __init__(self, handler, train_test="train"):
        self.handler = handler
        self.binary = train_test
        self.data, self.label = self.get_data_label()
    
    def __getitem__(self, index : int):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_data_label(self):
        data = []  # 存储数据集所有行的字符，index形式存储
        label = []  # 存储数据集所有行对应的标签

        if self.binary == "train":
            data = self.handler.train_arr
            label = self.handler.train_labels_arr
        else:
            data = self.handler.test_arr
            label = self.handler.test_labels_arr
        
        return data, label


# 在每个batch内填充0使得数据长度一致
def MyCollate(data):  # data是(input, label)的二元组，共有batch_size个
    crop_size = 499
    input_data = []
    input_label = []

    for d in data:
        start_idx = random.randint(0, len(d[0]) - crop_size)
        crop = d[0][start_idx: start_idx + crop_size]
        input_data.append(crop)
        input_label.append(d[1])

    input_data = np.array(input_data)
    input_data = input_data.astype(np.float32)
    input_data = torch.from_numpy(input_data)

    input_label = np.array(input_label)
    input_label = input_label.astype(np.float32)
    input_label = torch.from_numpy(input_label)

    return input_data, input_label


def LoadData(handler, shuffle=False, train_test="train", batch_size=16):
    dataset = TrailDataSet(handler=handler, train_test=train_test)
    dataloader = DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=MyCollate)
    return dataloader

if __name__ == "__main__":
    handler = File_handler()
    handler.load_data(adjective="bumpy", b_shuffle=True)
    dataset = TrailDataSet(handler=handler, train_test="train")
    dataloader = DataLoader(dataset=dataset, shuffle=True, batch_size=16, collate_fn=MyCollate)
    it = iter(dataloader)
    nex = next(it)
    print(nex)