'''
All rights reserved by Jerry
'''
import h5py
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
import pickle

Ep = ["SQUEEZE_SET_PRESSURE_SLOW", "HOLD_FOR_10_SECONDS", "SLIDE_5CM", "MOVE_DOWN_5CM"]
sensors = ["pac", "pdc", "tac", "tdc", "electrodes"]
# sensors = ["pac", "pdc", "tac", "tdc"]  # , "electrodes"]


class File_handler():
    def __init__(self, start_idx=0, end_idx=600, dataset_name='phac'):
        self.filename = './Data/PHAC/phac_train_test_pos_neg_90_10_1_20.hdf'
        self.dict_ep = {'DISABLED': 1, 'OPEN_GRIPPER_MAX': 2, 'MOVE_ARM_START_POSITION': 3,
                        'MOVE_GRIPPER_FAST_CLOSE': 4, 'OPEN_GRIPPER_BY_2CM_FAST': 5,
                        'FIND_CONTACT_CLOSE_GRIPPER_SLO': 6, 'SQUEEZE_SET_PRESSURE_SLOW': 7,
                        'CLOSE_GRIPPER_SLOW_TO_POSITION': 8, 'HOLD_FOR_10_SECONDS': 9,
                        'MOVE_UP_START_HEIGHT': 10, 'SLIDE_5CM': 11, 'OPEN_GRIPPER_FAST_2CM': 12,
                        'MOVE_UP_5CM': 13, 'MOVE_DOWN_5CM': 14, 'OPEN_GRIPPER_FAST_MAX': 15}

        self.dataset_name = 'phac'

        self.adjectives_mp = {}
        # self.keys_mp = {} #字典. map the trails of same object to same numbers。一个物体在key中最后一次出现的序号

        self.keys = []  # names of the trials, 每个物体每次数据采集的唯一标识符
        self.adjs = []  # dictionary of adjectives

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.n_adjs = 0

        self.n_trials = end_idx - start_idx

        self.signal_arr = []  # 读取并shuffle的信号数据，(n_Ep*n_sensor)*n*signal_size
        # self.labels_arr = [] #每组测试数据的目标物体的类别，n*1
        self.adjs_labels_arr = []  # 每组测试数据的形容词类别，*n*n_adjectives

        self.load_file()

    # 将所有形容词映射成数字
    def cal_label_adj(self, adj):
        '''
        map adjectives into label numbers
        '''
        l = len(adj)
        self.n_adjs = l
        for i in range(l):
            self.adjectives_mp[adj[i]] = i

    def load_file(self):
        '''
        load data , print informations and compute label numbers
        '''
        data = h5py.File(self.filename, 'r')
        self.data = data

        self.keys = list(data.keys())
        self.adjs = list(data['adjectives'].keys())

        # print information
        print("data of each object:")
        obj = data[self.keys[0]]
        '''for k in obj:
            print("  "+k)'''
        print("sensor data of each object:")
        for bio in obj["biotacs"]["finger_0"]:
            print("  " + bio)
        print("adjectives:")
        for a in self.adjs:
            print("  " + a)

        # 将两个非object的部分过滤掉
        print("loading data...")
        new_keys = []
        for k in self.keys:
            name = str(k)
            if (name.find("test") >= 0 or name.find("adject") >= 0):
                continue
            new_keys.append(k)

        self.cal_label_adj(self.adjs)

        self.keys = new_keys

        return data

    def export_csv(self, filaname='./adjectives.csv'):
        # 1. 创建文件对象
        f = open(filaname, 'w', newline="")
        # 2. 基于文件对象构建 csv写入对象
        csv_writer = csv.writer(f)
        # 3. 构建列表头
        csv_writer.writerow(["name"] + self.adjs)

        # 4. 写入csv文件内容
        raw_names, raw_adjs = self.get_info_data()

        # 根据文件名来索引
        for i in range(self.n_trials):
            name = self.keys[i]
            adj = raw_adjs[:, i]
            csv_writer.writerow([name] + adj.tolist())

        # 5. 关闭文件
        f.close()

    def get_info_data(self):
        '''
        get the names, labels of the names and objectives for each trial 
        * raw_name: n*1. unique for each trial
        * raw_labels: n*1. a label a class of the object. trials from same object have same labels
        * raw_adjs: n*n_adjs binary array. 
        * n is the number of trials, which is 600 for now
        '''
        raw_names = self.keys[self.start_idx:self.end_idx]
        # raw_labels = np.zeros(self.end_idx-self.start_idx, dtype=np.int32)
        raw_adjs = np.ones((self.n_adjs, self.end_idx - self.start_idx), dtype=np.int32) * -1
        for i in range(self.start_idx, self.end_idx):
            k = self.keys[i]
            # raw_labels[i-self.start_idx] = self.keys_mp[k[:-3]] #map the key to the object class

            obj = self.data[k]

            # 形容词
            for a in obj['adjectives']:
                a_str = a.decode('ascii')
                raw_adjs[self.adjectives_mp[a_str], i - self.start_idx] = 1

        return raw_names, raw_adjs

    def get_sensor_data(self, sensor_name='pac', explore_period='SLIDE_5CM'):
        '''
        get a specific kind of data from all objects given sensor name and explore period
        '''
        sensor_data_0 = []

        for i in range(self.start_idx, self.end_idx):
            k = self.keys[i]
            print(k)

            obj = self.data[k]

            biotac = obj['biotacs']
            finger_0 = biotac['finger_0']
            finger_1 = biotac['finger_1']
            sensor_data = np.array(finger_0[sensor_name])
            sensor_data = sensor_data - np.mean(sensor_data)

            # segmentation encode
            state = obj['state']
            state_detail = state['controller_detail_state']
            state_encoded = []

            # time_start = time.time() #开始计时

            # print(state_detail.dtype)
            # translate states into numbers
            '''for s in state_detail:  
                #state_encoded.append( self.dict_ep.get(s.decode('UTF-8'), 0) )
                state_encoded.append( self.dict_ep[s.decode('UTF-8')] )'''

            state_arr = np.array(state_detail)

            # time_end = time.time()    #结束计时
            # time_c = time_end - time_start   #运行所花时间
            # print('time cost for encoding controller state', time_c, 's')

            # extracting specific period
            # time_start = time.time()
            if len(explore_period) > 0:
                # EP_b = np.array(state_encoded) == self.dict_ep[explore_period]
                EP_b = state_arr == explore_period.encode()

                EP_idx = np.argwhere(EP_b)

                # segment sensor data according to EP
                sensor_data_ep = sensor_data[EP_b].reshape(-1)  # sensor_data特征的采样频率是其他的22倍
                sensor_data_0.append(sensor_data_ep)
            else:
                sensor_data_0.append(sensor_data.reshape(-1))
            # time_end = time.time()#结束计时

            # time_c= time_end - time_start   #运行所花时间
            # print('time cost for extracting specific period', time_c, 's')
        return sensor_data_0

    def shuffle_array(self, arrays_ori, shuffle_idx):
        arrays = []
        for i in shuffle_idx:
            arrays.append(arrays_ori[i])
        return arrays

    def load_data(self, b_shuffle=False, train_ratio=0.7):
        '''
        load data, shuffle each feature seperately
        split into training set and test set
        raw_*: 以trail为单位的数据
        *_arr: 以(n_Ep*n_sensors)为单位的数据
        '''

        names_arr = []
        adjs_labels_arr = []
        labels_arr = []
        signal_arr = []

        shuffle_idx = self.split_train_test(self.n_trials, train_ratio, b_shuffle=b_shuffle)
        raw_names, raw_adjs_labels = self.get_info_data()

        # infomations are same for different features in each trail
        names_shuffled = self.shuffle_array(raw_names, shuffle_idx)
        # labels_shuffled = raw_labels[shuffle_idx]
        adjs_labels_shuffled = raw_adjs_labels[:, shuffle_idx]

        # sparse_codes = np.zeros((280*4, self.n_trials)) #横轴是特征，纵轴是训练数据
        codes_idx = 0
        for ep in Ep:
            print("exploratory procedures : " + ep + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            for sensor in sensors:
                print("sensor : " + sensor + "=================================================>")
                signal_ori = self.get_sensor_data(sensor_name=sensor, explore_period=ep)

                # shuffle data and labels
                # shuffle indx is different for each feature
                signal_shuffled = self.shuffle_array(signal_ori, shuffle_idx)
                signal_arr.append(signal_shuffled)

                '''num_train = int(n*train_ratio)
                signal_train = signal[:num_train]
                signal_test = signal[num_train:]'''

        self.signal_arr = np.array(signal_arr)
        self.keys = np.array(names_shuffled)
        self.adjs_labels_arr = np.array(adjs_labels_shuffled)

    def split_train_test(self, n_trials=600, train_ratio=0.6, trial_each=10, b_shuffle=False):
        '''
        used before organized according to Ep and sensor names
        randomly split every 10 trials into training part and test part
        '''
        train_file = np.load('./Data/PHAC/split/train_split.npy', allow_pickle=True).item()
        test_file = np.load('./Data/PHAC/split/test_split.npy', allow_pickle=True).item()
        num_train = int(n_trials * train_ratio)

        idx_train, idx_test = self.load_train_test(load_path="./train_test/")
        if idx_train is None:
            # 每个物体收集trial_each次数据，确保每一类物体的数据都有测试和训练
            idx = np.reshape(range(n_trials), [-1, trial_each])
            n_obj = len(idx)
            idx_train = np.zeros([n_obj, int(trial_each * train_ratio)], dtype=np.int)
            idx_test = np.zeros([n_obj, int(trial_each * (1 - train_ratio))], dtype=np.int)

            train_each = int(trial_each * train_ratio)
            for i in range(n_obj):
                idx_obj = idx[i]
                if b_shuffle:
                    np.random.shuffle(idx_obj)
                idx_train[i, :] = idx_obj[:train_each]
                idx_test[i, :] = idx_obj[train_each:]

            idx_train = idx_train.flatten()
            idx_test = idx_test.flatten()
            if b_shuffle:
                np.random.shuffle(idx_train)
                np.random.shuffle(idx_test)

            self.save_train_test(idx_train, idx_test)

        shuffle_idx = np.concatenate([idx_train.flatten(), idx_test.flatten()])

        # infomations are same for different features in each trail
        # self.signal_arr = self.signal_arr[:,shuffle_idx]
        # self.labels_arr = self.labels_arr[shuffle_idx]
        # self.adjs_labels_arr = self.adjs_labels_arr[:,shuffle_idx]
        # self.raw_data = self.shuffle_array(raw_data, shuffle_idx)
        # self.gestures = self.shuffle_array(gestures, shuffle_idx)
        # self.keys = self.shuffle_array(keys, shuffle_idx)

        return shuffle_idx

    def save_train_test(self, idx_train, idx_test, save_path="./Data/train_test/"):
        self.save_to_pickle(idx_train, save_path + self.dataset_name + str(self.n_trials) + '_train_idx')
        self.save_to_pickle(idx_test, save_path + self.dataset_name + str(self.n_trials) + '_test_idx')

    def load_train_test(self, load_path="./train_test/"):
        idx_train = self.read_from_pickle(load_path + self.dataset_name + str(self.n_trials) + '_train_idx')
        idx_test = self.read_from_pickle(load_path + self.dataset_name + str(self.n_trials) + '_test_idx')

        return idx_train, idx_test

    def read_from_pickle(self, path):
        import os
        if not os.path.exists(path):
            return None

        with open(path, "rb+") as fk:
            s = fk.read()
            data = pickle.loads(s)
            return data

        return None

    def save_to_pickle(self, data, path):
        with open(path, "wb+") as fs:
            f = pickle.dumps(data)
            fs.write(f)
            fs.close()

    def get_shuffle_idx_1(self, n_trials=600, train_ratio=0.6):
        '''
        shuffle data by column first
        shuffle data by row next
        '''
        # 共60种物体，每个物体收集10次数据，确保每一类物体的数据都有测试和训练
        idx = np.reshape(range(n_trials), [-1, 10])

        # shuffle in all training sets and test sets
        r_tr = int(10 * train_ratio)  # 60%用于训练

        # 确保每一类物体的数据都有测试和训练
        train_idx = idx[:, :r_tr].reshape(-1)
        test_idx = idx[:, r_tr:].reshape(-1)
        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)
        shuffle_idx = np.concatenate([train_idx, test_idx], axis=0)

        return shuffle_idx

    def get_shuffle_idx_2(self, n_trials=600, n_trials_each=10, train_ratio=0.6):
        '''
        shuffle data for each kind of object.
        注意每种物体都有训练数据和测试数据
        '''
        # 共60种物体，每个物体收集10次数据，确保每一类物体的数据都有测试和训练
        shuffle_idx = np.reshape(range(n_trials), [-1, 10])
        n_class = n_trials // n_trials_each
        for i in range(n_class):
            idx_trials = np.arange(10)
            np.random.shuffle(idx_trials)  # shuffle by 10 trials
            shuffle_idx[i, :] = shuffle_idx[i, idx_trials]

        # 将训练数据和测试数据分开，方便取出
        trials_train = int(n_trials * train_ratio)
        train_idx = shuffle_idx[:, :trials_train].reshape(-1)
        test_idx = shuffle_idx[:, trials_train:].reshape(-1)

        return np.concatenate([train_idx, test_idx], axis=0)


# 线性核的支持向量机
def plot_point_linear(dataArr, labelArr, Support_vector_index, W, b):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='r', s=20)

    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=50, c='', alpha=0.5, linewidth=1.5, edgecolor='red')

    x = np.arange(0, 50, 0.1)
    y = (W[0][0] * x + b) / -W[0][1]
    plt.scatter(x, y, s=3, marker='h')
    plt.show()


# 不绘制边界的支持向量机
def plot_point(dataArr, labelArr, Support_vector_index=[]):
    for i in range(np.shape(dataArr)[0]):
        if labelArr[i] == 1:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='b', s=20)
        else:
            plt.scatter(dataArr[i][0], dataArr[i][1], c='r', s=20)

    for j in Support_vector_index:
        plt.scatter(dataArr[j][0], dataArr[j][1], s=50, c='', alpha=0.5, linewidth=1.5, edgecolor='red')

    plt.show()


# 绘制傅里叶变换结果
def plot_fft(y, yf, name=""):
    x = np.arange(len(y))
    x_fft = np.arange(len(yf))

    # 绘制
    plt.figure()
    plt.subplot(211)
    plt.plot(x, y)
    plt.title('Original PAC ' + name)

    plt.subplot(212)
    plt.plot(x_fft, yf, 'b')
    plt.title('FFT of Mixed wave(two sides frequency range)', fontsize=9, color='r')

    plt.show()


def IO_write(file_name, feature_train, feature_test):
    with open(file_name, 'w') as f:  # 默认模式为‘r’，只读模式
        feature_len = feature_train.shape[0]
        n_train = feature_train.shape[1]
        n_test = feature_test.shape[1]
        f.write("{} {} {}".format(feature_len, n_train, n_test))

        for i in range(n_train):
            f.write(feature_train[i, :])
        for i in range(n_test):
            f.write(feature_test[i, :])


def IO_read(file_name):
    with open(file_name) as f:  # 默认模式为‘r’，只读模式
        feature_len, n_train, n_test = f.readline()  # 读取文件全部内容
        feature_train = np.zeros((feature_len, n_train))
        feature_test = np.zeros((feature_len, n_test))
        for i in range(n_train):
            feature_train[i, :] = f.readline()

        for i in range(n_test):
            feature_test[i, :] = f.readline()

    return feature_train, feature_test


if __name__ == "__main__":
    handler = File_handler(end_idx=600)
    # handler.export_csv()
    handler.load_data()
    data = handler.get_sensor_data(explore_period="")[0]
    data_ep = handler.get_sensor_data(explore_period="SQUEEZE_SET_PRESSURE_SLOW")[0]
    data_ep2 = handler.get_sensor_data(explore_period="SLIDE_5CM")[0]
    idx = idx2 = 0
    for i in range(len(data) - len(data_ep) - 20):
        b_same = True
        for t in range(10):
            if data[i + t * 2] != data_ep[t * 2]:
                b_same = False
                break
        if (b_same):
            idx = i
            break
    for i in range(len(data) - len(data_ep2) - 20):
        b_same = True
        for t in range(10):
            if data[i + t * 2] != data_ep2[t * 2]:
                b_same = False
                break
        if (b_same):
            idx2 = i
            break

    plt.plot(range(len(data)), data, 'r')
    plt.plot(range(idx, idx + len(data_ep)), data_ep, 'g')
    plt.plot(range(idx2, idx2 + len(data_ep2)), data_ep2, 'g')
    plt.show()
