# 读取label-test文件，生成每条数据对应的形容词标签

import os 
import csv
import numpy as np

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

# 根据label_dict字典及滑动数据记录的材质，为每条滑动数据添加形容词取值（标签）
path = "./DataProcess/Data/npydata12"
materials = os.listdir(path)
dataset = {}  # 按照每种材质的类别存储文件和标签的对应关系
# 记录数据集中的每条滑动数据
for material in materials:  # 获取当前数据对应的材质
    material_path = path + '/' + material
    files = os.listdir(material_path)
    data = [[] for _ in range(len(files))]
    for i, file in enumerate(files):
        data[i].append(file)  # 每行数据第一个元素为文件名
        # for k in label_dict[material]:# 根据材质添加形容词取值情况
        for k in label_dict.get(material, []):  
            data[i].append(k)
    dataset[material] = data
print(dataset)  # 最终形式为 材质类别：数据文件名+对应的形容词标签

save_data = np.array(dataset)
np.save("./DataProcess/Data/datalabel12", save_data)

# 根据label_dict字典及滑动数据记录的材质，为每条滑动数据（小波变化处理后）添加形容词取值（标签）
file = os.listdir("./DataProcess/Data/wavedata12")
materials = os.listdir(path)
dataset = {}  # 按照每种材质的类别存储文件和标签的对应关系
# 记录数据集中的每条滑动数据
for material in materials:  # 获取当前数据对应的材质
    material_path = path + '/' + material
    files = os.listdir(material_path)
    data = [[] for _ in range(len(files))]
    for i, file in enumerate(files):
        data[i].append(file)  # 每行数据第一个元素为文件名
        for k in label_dict.get(material, []):
        # for k in label_dict[material]:  # 根据材质添加形容词取值情况
            data[i].append(k)
    dataset[material] = data
print(dataset)  # 最终形式为 材质类别：数据文件名+对应的形容词标签

save_data = np.array(dataset)
np.save("./DataProcess/Data/wavelabel12", save_data)