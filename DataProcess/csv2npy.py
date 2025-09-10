# 用于将原始csv格式文件提取有效信息并生成npy格式文件

import os
import csv
import numpy as np
from matplotlib import pyplot as plt


def generate_data(name):
    # 读取csv文件
    print(name)
    f = open(name, 'r')
    reader = csv.reader(f)

    # 存储压力、执行的动作和imu数据
    data_length = 1000
    data = []
    action = []
    
    # 一行一行地读取数据
    for i in reader:
        if len(i) == 17:
            action.append(int(float(i[13])))
            data.append(list(map(float, i)))


    # 区分快划和慢划 [采集过程为：按压->保持->快划->悬空（返回）->按压->保持->慢划->悬空（返回）,3->4->5->2->3->4->5->2->3]
    def get_state_info(action):
        state = action[0]  # 记录初始状态
        index_save = [0]
        for i in range(len(action)):
            # action变化说明触觉传感器的状态发生变化
            if action[i] != state:
                # print(i)
                state = action[i]
                index_save.append(i)  # 记录状态发生变化的结点
        # print(index_save)
        squeeze = [index_save[2] - data_length, index_save[2]]
        hold = [index_save[2], index_save[3]]
        fast_slide = [index_save[3], index_save[4]]
        slow_slide = [index_save[7], index_save[8]]
        return squeeze, hold, fast_slide, slow_slide


    sque, hold, fast, slow = get_state_info(action)
    # print(sque, hold, fast, slow)

    # p5随快划慢划的时间变化曲线
    # for i in range(len(fast)):
    #     x_f = np.arange(0, fast[i][1] - fast[i][0])
    #     y_f = np.array(p5[fast[i][0]:fast[i][1]])
    #     x_s = np.arange(0, slow[i][1] - slow[i][0])
    #     y_s = np.array(p5[slow[i][0]:slow[i][1]])
    #     ax1 = plt.subplot(2, len(fast), 1 + i)
    #     ax1.plot(x_f, y_f)
    #     ax2 = plt.subplot(2, len(fast), len(fast) + 1 + i)
    #     ax2.plot(x_s, y_s)
    # plt.show()

    # 取快划和慢划的中间1000帧数据并保存 [中间1000帧数据较稳定]
    sque_mid = round((sque[1] + sque[0]) / 2)  # 计算快划数据的中间帧
    hold_mid = round((hold[1] + hold[0]) / 2)  # 计算慢划数据的中间帧
    fast_mid = round((fast[1] + fast[0]) / 2)  # 计算快划数据的中间帧
    slow_mid = round((slow[1] + slow[0]) / 2)  # 计算慢划数据的中间帧
    
    # 从中间帧往前后推500帧获得中间1000帧
    data = np.array(data)
    data_1000_sque = data[hold[1] - 1000: hold[1]]
    data_1000_hold = data[hold_mid - data_length // 2: hold_mid + data_length // 2]
    data_1000_fast = data[fast_mid - data_length // 2: fast_mid + data_length // 2]
    data_1000_slow = data[slow_mid - data_length // 2: slow_mid + data_length // 2]
    
    # 数据归一化
    for i in range(len(data_1000_sque[0])):
        data_1000_sque[:, i] = ((data_1000_sque[:, i] - np.mean(data_1000_sque[:, i])) / np.std(data_1000_sque[:, i]) if i != 13 else data_1000_sque[:, i])
    for i in range(len(data_1000_hold[0])):
        data_1000_hold[:, i] = ((data_1000_hold[:, i] - np.mean(data_1000_hold[:, i])) / np.std(data_1000_hold[:, i]) if i != 13 else data_1000_hold[:, i])
    for i in range(len(data_1000_fast[0])):
        data_1000_fast[:, i] = ((data_1000_fast[:, i] - np.mean(data_1000_fast[:, i])) / np.std(data_1000_fast[:, i]) if i != 13 else data_1000_fast[:, i])
    for i in range(len(data_1000_slow[0])):
        data_1000_slow[:, i] = ((data_1000_slow[:, i] - np.mean(data_1000_slow[:, i])) / np.std(data_1000_slow[:, i]) if i != 13 else data_1000_slow[:, i])

    # 最终要保存的数据，按压、保持、快划和慢划各一行
    save_data = np.array([
        # [data_1000_sque[:, 1], data_1000_sque[:, 2], data_1000_sque[:, 3], data_1000_sque[:, 4], data_1000_sque[:, 5], data_1000_sque[:, 6], data_1000_sque[:, 7], data_1000_sque[:, 8],
        # data_1000_sque[:, 9], data_1000_sque[:, 10], data_1000_sque[:, 11], data_1000_sque[:, 12], data_1000_sque[:, 14], data_1000_sque[:, 15], data_1000_sque[:, 16]],
        # [data_1000_hold[:, 1], data_1000_hold[:, 2], data_1000_hold[:, 3], data_1000_hold[:, 4], data_1000_hold[:, 5], data_1000_hold[:, 6], data_1000_hold[:, 7], data_1000_hold[:, 8],
        # data_1000_hold[:, 9], data_1000_hold[:, 10], data_1000_hold[:, 11], data_1000_hold[:, 12], data_1000_hold[:, 14], data_1000_hold[:, 15], data_1000_hold[:, 16]],
        [data_1000_fast[:, 1], data_1000_fast[:, 2], data_1000_fast[:, 3], data_1000_fast[:, 4], data_1000_fast[:, 5], data_1000_fast[:, 6], data_1000_fast[:, 7], data_1000_fast[:, 8],
        data_1000_fast[:, 9], data_1000_fast[:, 10], data_1000_fast[:, 11], data_1000_fast[:, 12], data_1000_fast[:, 14], data_1000_fast[:, 15], data_1000_fast[:, 16]],
        [data_1000_slow[:, 1], data_1000_slow[:, 2], data_1000_slow[:, 3], data_1000_slow[:, 4], data_1000_slow[:, 5], data_1000_slow[:, 6], data_1000_slow[:, 7], data_1000_slow[:, 8],
        data_1000_slow[:, 9], data_1000_slow[:, 10], data_1000_slow[:, 11], data_1000_slow[:, 12], data_1000_slow[:, 14], data_1000_slow[:, 15], data_1000_slow[:, 16]]
        ])

    # np.save(name[:-4]+"_%s"%i,save_data)
    name = name.split('/')  # [., Data, tactiledata12, Aluminium, Aluminium_1.csv]
    if not os.path.exists("./Data/npydata12/" + name[3]):
        os.mkdir("./Data/npydata12/" + name[3])
    np.save("./Data/npydata12/" + name[3] + '/' + name[4][:-4], save_data)

    # p_total随快划慢划（中间1000帧）的时间变化曲线
    # x = np.arange(0, 1000)
    # ax = plt.subplot(2, len(f), 1 + i)
    # ax.plot(x, ptotal_1000_fast)
    # ax2 = plt.subplot(2, len(f), len(f) + 1 + i)
    # ax2.plot(x, ptotal_1000_slow)
    # plt.savefig('p_total的快划慢划变化曲线.jpg')


if __name__ == "__main__":
    path = "./Data/tactiledata12"
    materials = os.listdir(path)
    if not os.path.exists("./Data/npydata12"):
        os.mkdir("./Data/npydata12")
    # print(names)
    for material in materials:
        material_path = path + '/' + material
        files = os.listdir(material_path)
        for file in files:
            file_path = material_path + '/' + file  # "./Data/tactiledata12/Aluminium/Aluminium_1.csv"
            generate_data(file_path)
