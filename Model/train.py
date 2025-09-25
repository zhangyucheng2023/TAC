# -*- coding: utf-8 -*-
# author:Yuanpei Zhang
# 数据集的训练与测试

import sys
import numpy as np
import torch
import os
from torch import nn
import torch.utils.data as Data
from thop import profile, clever_format
from torchstat import stat
from torchsummaryX import summary
from ptflops import get_model_complexity_info
from conv1D import convolution1d
from conv2D import convolution2d
from MLP import fullconnection
from LSTM import lstm_v1, lstm_v2
from Attention import atten
from AttentionLSTM import lstm_atten_v1, lstm_atten_v2
from Attentionconv1D import convolution1d_atten
from utils import TensorValue2d_to_PredictLabel, Accuracy_Sum, F1_Score_Sum

# 选择使用的网络模型
print("请选择使用的网络模型："
        "1.一维卷积 "
        "2.二维卷积 "
        "3.多层感知机 "
        "4.单向LSTM "
        "5.双向LSTM "
        "6.Attention+单向LSTM "
        "7.Attention+双向LSTM "
        "8.Attention+一维卷积 "
        "9.Attention ")
choose = input()

if choose == "1":
    # 设置训练参数
    net = convolution1d(embedding_dim=30)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 4e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3
    INPUT_SIZE = (2, 15, 1000)

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/conv1D/"

elif choose == "2":
    # 设置训练参数
    net = convolution2d()
    BATCH_SIZE = 16
    NUM_EPOCHS = 100  # w/ BN 80, w/o BN 100
    LEARNING_RATE = 2e-4
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 2
    INPUT_SIZE = (4, 120, 200)

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/WaveDataset12/"
    MODEL_PATH = "./Model/models/conv2D/"

elif choose == "3":
    # 设置训练参数
    net = fullconnection()
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    LR_DECAY = False
    STEP_SIZE = NUM_EPOCHS / 2
    INPUT_SIZE = (2, 15, 1000)

    # 设置数据路径和保存模型路径
    DATA_PATH = "./Data/Dataset12/"
    MODEL_PATH = "./Model/models/MLP/"

elif choose == "4":
    # 设置训练参数
    net = lstm_v1(embedding_dim=30, hidden_dim=15)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3
    INPUT_SIZE = (2, 15, 1000)

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/LSTM_v1/"

elif choose == "5":
    # 设置训练参数
    net = lstm_v2(embedding_dim=30, hidden_dim=15)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3
    INPUT_SIZE = (2, 15, 1000)

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/LSTM_v2/"

elif choose == "6":
    # 设置训练参数
    net = lstm_atten_v1(hidden_dim=30, num_layers=3)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/AttentionLSTM_v1/"

elif choose == "7":
    # 设置训练参数
    net = lstm_atten_v2(embedding_dim=30, hidden_dim=15)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 3e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/AttentionLSTM_v2/"

elif choose == "8":
    # 设置训练参数
    net = convolution1d_atten(embedding_dim=30)
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 5e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/Attentionconv1D/"

elif choose == "9":
    # 设置训练参数
    net = atten(embedding_dim=30)
    # BATCH_SIZE = 16
    # NUM_EPOCHS = 80  # w/ BN 60, w/o BN 80
    # LEARNING_RATE = 5e-4
    # LR_DECAY = True
    # STEP_SIZE = NUM_EPOCHS / 2
    BATCH_SIZE = 16
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3
    LR_DECAY = True
    STEP_SIZE = NUM_EPOCHS / 3

    # 设置数据路径和保存模型路径
    DATA_PATH = "./DataProcess/Data/Dataset12/"
    MODEL_PATH = "./Model/models/Attention/"

else:
    print("您输入的网络模型不存在！")
    sys.exit()


# 加载训练数据
train_feature = np.load(DATA_PATH + "train_data.npy")
train_label = np.load(DATA_PATH + "train_label.npy")
# train_label = train_label[:, 1:]  # 去除最开始的文件名
train_label = train_label.astype(np.float32)
train_feature = train_feature.astype(np.float32)
train_label = torch.from_numpy(train_label)
train_feature = torch.from_numpy(train_feature)

train_dataset = Data.TensorDataset(train_feature, train_label)
train_dataloader = Data.DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

# 加载测试数据
test_feature = np.load(DATA_PATH + "test_data.npy")
test_label = np.load(DATA_PATH + "test_label.npy")
# test_label = test_label[:, 1:]
test_label = test_label.astype(np.float32)
test_feature = test_feature.astype(np.float32)
test_label = torch.from_numpy(test_label)
test_feature = torch.from_numpy(test_feature)

test_dataset = Data.TensorDataset(test_feature, test_label)
test_dataloader = Data.DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

# 初始化模型
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(DEVICE)
print(net)

# 设置优化器、损失函数
pos_weight = [2, 3, 0.714285714, 5, 99, 99, 99, 3, 23, 99, 99, 1.4, 99, 11, 0.714285714, 11, 0.846153846, 1.666666667, 3.8, 99, 2.428571429, 0.6, 5, 99]
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE)
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
# criterion = nn.MultiLabelSoftMarginLoss()
criterion = criterion.to(DEVICE)

# 模型的训练与测试
max_acc, max_pre, max_rec, max_f1 = 0, 0, 0, 0
for epoch in range(NUM_EPOCHS):
    # 每个epoch训练
    net.train()
    train_loss = 0
    train_correct = 0
    test_correct = 0
    for train_batches, (input, label) in enumerate(train_dataloader):    
        # 计算训练损失
        input = torch.as_tensor(input, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.float32)
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        output = net(input)
        loss = criterion(output, label)
        train_loss += loss.item()

        # 误差反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算训练集准确率
        with torch.no_grad():
            predict = TensorValue2d_to_PredictLabel(output)
            train_correct += Accuracy_Sum(predict, label)
    
    # 学习率衰减
    if LR_DECAY:
        scheduler.step()

    # 每个epoch测试
    net.eval()
    test_correct = 0
    TP = FP = TN = FN = 0
    for input, label in test_dataloader:
        input = torch.as_tensor(input, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.float32)
        input = input.to(DEVICE)
        label = label.to(DEVICE)
        output = net(input)
        output = TensorValue2d_to_PredictLabel(output)
        test_correct += Accuracy_Sum(output, label)
        batch = output.shape[0]
        adj = output.shape[1]
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
        
    # 打印每个epoch的训练损失、训练准确率、测试准确率
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP + 1e-8)  # 防止分母取值为0
    recall = TP / (TP + FN + 1e-8)  # 防止分母取值为0
    f1_score = 2 * precision * recall / (precision + recall + 1e-8)

    print("Epoch: [%d/%d]  Train_Loss: %.3f  Train_Acc:%.3f  Test_Acc:%.4f" % (
    epoch + 1, NUM_EPOCHS, train_loss / (train_batches + 1),
    train_correct / len(train_dataset), test_correct / len(test_dataset)))
    print("Epoch: [%d/%d]  TP: %d  FP:%d  FN:%d  TN:%d" % (epoch + 1, NUM_EPOCHS, TP, FP, FN, TN))
    print("Epoch: [%d/%d]  Accuracy: %.4f  Precision:%.4f  Recall:%.4f  F1-Score:%.4f" % (
        epoch + 1, NUM_EPOCHS, accuracy, precision, recall, f1_score))
    print()  # 输出的每个Epoch之间有一个空行

    # 每20个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        path = MODEL_PATH + "net_%s.pt" % (epoch + 1)
        directory = os.path.dirname(path) # 确保目录存在
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(net.state_dict(), path)

    max_acc = max(max_acc, accuracy)
    if f1_score > max_f1:
        max_f1 = f1_score
        max_pre = precision
        max_rec = recall
    # max_f1 = max(max_f1, f1_score)

macs, params = get_model_complexity_info(net.to(torch.device("cpu")), (4, 120, 200) if choose == "2" else (2, 15, 1000), print_per_layer_stat=True)
print("{:<30}  {:<8}".format("Computational complexity: ", macs))
print("{:<30}  {:<8}".format("Number of parameters: ", params))
# flops, params = profile(net, (input,))
# macs, params = clever_format([flops, params], "%.3f")
# print('MACs: ', macs, 'FLOPs: ', flops, 'params: ', params)
# stat(net.to(torch.device("cpu")), (2, 15, 1000))
# summary(net, input)
print("\nMax_Acc: %.4f  Max_Pre: %.4f  Max_Rec: %.4f  Max_F1: %.4f\n" % (max_acc, max_pre, max_rec, max_f1))