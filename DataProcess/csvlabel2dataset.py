# 根据文件名和标签的对应关系，为每条滑动数据打标，形成数据集，并随机排序划分出train和validation集

import os
import csv
import copy
import numpy as np


# 随机划分数据和标签
def random_shuffle(data, label):
    randnum = 999  # 保证data和label的打乱顺序一致
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data, label


def SeenDivide(label_dict, train_ratio=0.8):  # label_dict材质与形容词的对应关系，train_ratio训练集占比
    # 划分原始触觉数据集
    # 读取每条触觉数据与形容词的对应关系
    # label_dict = np.load("Data/datalabel12.npy", allow_pickle=True).item()
    # print(label['ArtificialStone'])

    # 制作数据集
    # 1用于后续数据之间的串联，2对应四种数据采集状态，15为每组数据的维数，1000代表1000帧，24为形容词个数
    concated_data = np.zeros([1, 2, 15, 1000])
    concated_label = np.zeros([1, 24])
    path = "./DataProcess/Data/npydata12"
    materials = os.listdir(path)
    for material in materials:
        # label = label_dict[material]
        label = label_dict.get(material, np.zeros(24))
        if label.size == 0:
            print(f"Warning: Material '{material}' has an empty label.")
            continue
        
        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            concated_data = np.concatenate((concated_data, data), axis=0)  # 将数据串联在一起
            concated_label = np.concatenate((concated_label, label), axis=0)  # 将标签串联在一起
    concated_data = concated_data[1:, :, :, :]  # 去除初始的全零块，N*2*15*100
    concated_label = concated_label[1:, :]  # 去除初始的全零块，N*24
    # print(concated_data.shape)

    # 划分训练集和验证集
    D, L = random_shuffle(concated_data, concated_label)
    split_index = round(concated_label.shape[0] * train_ratio)  # 前split_index个数据和标签为训练集
    train_data = D[:split_index, :, :, :]
    train_label = L[:split_index, :]
    test_data = D[split_index:, :, :, :]
    test_label = L[split_index:, :]
    print(train_data.shape)
    print(test_data.shape)
    # 存储训练集和验证集
    np.save("./DataProcess/Data/Dataset12/train_data.npy", train_data)
    np.save("./DataProcess/Data/Dataset12/train_label.npy", train_label)
    np.save("./DataProcess/Data/Dataset12/test_data.npy", test_data)
    np.save("./DataProcess/Data/Dataset12/test_label.npy", test_label)
    

    # 划分小波变换后的触觉数据集
    # 读取数据（小波变换处理后）
    # label_dict = np.load("Data/wavelabel12.npy", allow_pickle=True).item()
    # print(label['ArtificialStone'])

    # 制作数据集（小波变换后）
    # 1用于后续数据之间的串联，小波变换后，数据的维度为4*120*200
    concated_data = np.zeros([1, 4, 120, 200])
    concated_label = np.zeros([1, 24])
    path = "./DataProcess/Data/wavedata12"  # 由于wavellabel12是由wavedata12生成的，因此可以保证数据和标签的对应关系
    materials = os.listdir(path)
    for material in materials:
        # label = label_dict[material]
        label = label_dict.get(material, None)  # 如果 material 不在 label_dict 中，则返回 None
        if label is None:
            print(f"Warning: Material '{material}' not found in label_dict.")
            continue  # 或者根据需要处理这种情况

        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            concated_data = np.concatenate((concated_data, data), axis=0)  # 将数据串联在一起
            concated_label = np.concatenate((concated_label, label), axis=0)  # 将标签串联在一起
    concated_data = concated_data[1:, :, :, :]  # 去除初始的全零块，N*4*120*200
    concated_label = concated_label[1:, :]  # 去除初始的全零块，N*24
    # print(concated_data.shape)

    # 划分训练集和验证集
    D, L = random_shuffle(concated_data, concated_label)
    split_index = round(concated_label.shape[0] * train_ratio)  # 前split_index个数据和标签为训练集
    train_data = D[:split_index, :, :, :]
    train_label = L[:split_index, :]
    test_data = D[split_index:, :, :, :]
    test_label = L[split_index:, :]
    print(train_data.shape)
    print(test_data.shape)
    # 存储训练集和验证集
    np.save("./DataProcess/Data/WaveDataset12/train_data.npy", train_data)
    np.save("./DataProcess/Data/WaveDataset12/train_label.npy", train_label)
    np.save("./DataProcess/Data/WaveDataset12/test_data.npy", test_data)
    np.save("./DataProcess/Data/WaveDataset12/test_label.npy", test_label)


def UnseenDivide(label_dict, train_ratio=0.8):  # label_dict材质与形容词的对应关系，train_ratio训练集占比
    # 读取每条触觉数据与形容词的对应关系
    # label = np.load("Data/datalabel12.npy")
    # print(label.shape[0])

    # 划分训练集和验证集对应的材质
    path = "./Data/npydata12"
    materials = os.listdir(path)
    np.random.shuffle(materials)
    test_materials = ["CardBoard"]
    train_materials = copy.deepcopy(materials)
    train_materials.remove(test_materials[0])
    # split_index = round(len(materials) * train_ratio)
    # train_materials = materials[:split_index]  # 前split_index种材质为训练集
    # test_materials = materials[split_index:]

    # 划分原始触觉数据集
    # 生成训练集
    # 1用于后续数据之间的串联，2对应四种数据采集状态，15为每组数据的维数，1000代表1000帧，24为形容词个数
    train_data = np.zeros([1, 2, 15, 1000])
    train_label = np.zeros([1, 24])
    for material in train_materials:
        label = label_dict[material]
        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            train_data = np.concatenate((train_data, data), axis=0)  # 将数据串联在一起
            train_label = np.concatenate((train_label, label), axis=0)  # 将标签串联在一起
    train_data = train_data[1:, :, :, :]  # 去除初始的全零块，N*2*15*100
    train_label = train_label[1:, :]  # 去除初始的全零块，N*24
    np.save("./Data/Dataset12/train_data.npy", train_data)
    np.save("./Data/Dataset12/train_label.npy", train_label)
    print(train_data.shape)

    # 生成验证集
    # 1用于后续数据之间的串联，2对应四种数据采集状态，15为每组数据的维数，1000代表1000帧，24为形容词个数
    test_data = np.zeros([1, 2, 15, 1000])
    test_label = np.zeros([1, 24])
    for material in test_materials:
        label = label_dict[material]
        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            test_data = np.concatenate((test_data, data), axis=0)  # 将数据串联在一起
            test_label = np.concatenate((test_label, label), axis=0)  # 将标签串联在一起
    test_data = test_data[1:, :, :, :]  # 去除初始的全零块，N*2*15*100
    test_label = test_label[1:, :]  # 去除初始的全零块，N*24
    np.save("./Data/Dataset12/test_data.npy", test_data)
    np.save("./Data/Dataset12/test_label.npy", test_label)
    print(test_data.shape)


    # 划分小波变换后的触觉数据集
    path = "./Data/wavedata12"

    # 生成训练集
    # 1用于后续数据之间的串联，小波变换后，数据的维度为4*120*200
    train_data = np.zeros([1, 4, 120, 200])
    train_label = np.zeros([1, 24])
    for material in train_materials:
        label = label_dict[material]
        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            train_data = np.concatenate((train_data, data), axis=0)  # 将数据串联在一起
            train_label = np.concatenate((train_label, label), axis=0)  # 将标签串联在一起
    train_data = train_data[1:, :, :, :]  # 去除初始的全零块，N*4*120*200
    train_label = train_label[1:, :]  # 去除初始的全零块，N*24
    np.save("./Data/WaveDataset12/train_data.npy", train_data)
    np.save("./Data/WaveDataset12/train_label.npy", train_label)
    print(train_data.shape)

    # 生成验证集
    # 1用于后续数据之间的串联，小波变换后，数据的维度为4*120*200
    test_data = np.zeros([1, 4, 120, 200])
    test_label = np.zeros([1, 24])
    for material in test_materials:
        label = label_dict[material]
        label = np.expand_dims(label, 0)
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            # print(i)
            data = np.load(material_path + '/' + file)
            data = np.expand_dims(data, 0)
            test_data = np.concatenate((test_data, data), axis=0)  # 将数据串联在一起
            test_label = np.concatenate((test_label, label), axis=0)  # 将标签串联在一起
    test_data = test_data[1:, :, :, :]  # 去除初始的全零块，N*4*120*200
    test_label = test_label[1:, :]  # 去除初始的全零块，N*24
    np.save("./Data/WaveDataset12/test_data.npy", test_data)
    np.save("./Data/WaveDataset12/test_label.npy", test_label)
    print(test_data.shape)


# 读取labels-test.csv文件
f = open("./DataProcess/Data/labels-test.csv", 'r', encoding='gbk')
reader = csv.reader(f)
data = []
sample_num = 19  # 19种材质（光滑木板、胶合板等）
for i, j in enumerate(reader):
    if 1 < i < 21:  # csv文件的2-20行，共19种材质
        data.append(j)

# 将每种材质及其对应的每个形容词取值存储为label_dict字典
label_dict = {}
for i in range(sample_num):  # 对于每种材质
    label_dict[data[i][1]] = [float(j) for j in data[i][3:]]  # 存储其形容词取值情况
# print(label_dict)
# print(label_dict['Rubber'])

SeenDivide(label_dict)