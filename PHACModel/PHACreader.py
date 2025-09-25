'''
All rights reserved by Jerry and Yuanpei Zhang
'''
import h5py
import numpy as np


Eps = ["SQUEEZE_SET_PRESSURE_SLOW", "HOLD_FOR_10_SECONDS", "SLIDE_5CM", "MOVE_DOWN_5CM"]
Sensors = ["pac", "pdc", "tac", "tdc", "electrodes"]
Adjectives = ["absorbent", "bumpy", "compressible", "cool", "crinkly", "fuzzy", "hairy", "hard",
            "metallic", "nice", "porous", "rough", "scratchy", "slippery", "smooth", "soft",
            "solid", "springy", "squishy", "sticky", "textured", "thick", "thin", "unpleasant"]


class File_handler():
    def __init__(self, start_idx=0, end_idx=600, dataset_name='phac'):
        self.filename = './DataProcess/Data/PHAC/phac_train_test_pos_neg_90_10_1_20.hdf'
        self.dict_ep = {'DISABLED': 1, 'OPEN_GRIPPER_MAX': 2, 'MOVE_ARM_START_POSITION': 3,
                        'MOVE_GRIPPER_FAST_CLOSE': 4, 'OPEN_GRIPPER_BY_2CM_FAST': 5,
                        'FIND_CONTACT_CLOSE_GRIPPER_SLO': 6, 'SQUEEZE_SET_PRESSURE_SLOW': 7,
                        'CLOSE_GRIPPER_SLOW_TO_POSITION': 8, 'HOLD_FOR_10_SECONDS': 9,
                        'MOVE_UP_START_HEIGHT': 10, 'SLIDE_5CM': 11, 'OPEN_GRIPPER_FAST_2CM': 12,
                        'MOVE_UP_5CM': 13, 'MOVE_DOWN_5CM': 14, 'OPEN_GRIPPER_FAST_MAX': 15}

        self.dataset_name = 'phac'

        self.adjectives_mp = {}

        self.keys = []  # names of the trials, 每个物体每次数据采集的唯一标识符
        self.adjs = []  # dictionary of adjectives

        self.start_idx = start_idx
        self.end_idx = end_idx

        self.n_adjs = 0

        self.n_trials = end_idx - start_idx

        # self.signal_arr = []  # 读取并shuffle的信号数据，(n_Ep*n_sensor)*n*signal_size
        # self.adjs_labels_arr = []  # 每组测试数据的形容词类别，*n*n_adjectives
        self.train_arr = []
        self.train_labels_arr = []
        self.test_arr = []
        self.test_labels_arr = []

        self.load_file()

    def load_file(self):
        '''
        load data , print informations and compute label numbers
        '''
        data = h5py.File(self.filename, 'r')
        self.data = data

        self.keys = list(data.keys())
        self.adjs = list(data['adjectives'].keys())

        # print information
        # print("data of each object:")
        # obj = data[self.keys[0]]
        # '''for k in obj:
        #     print("  "+k)'''
        # print("sensor data of each object:")
        # for bio in obj["biotacs"]["finger_0"]:
        #     print("  " + bio)
        # print("adjectives:")
        # for a in self.adjs:
        #     print("  " + a)

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

    # 将所有形容词映射成数字
    def cal_label_adj(self, adj):
        '''
        map adjectives into label numbers
        '''
        l = len(adj)
        self.n_adjs = l
        for i in range(l):
            self.adjectives_mp[adj[i]] = i

    def load_data(self, adjective="absorbent", ep="SQUEEZE_SET_PRESSURE_SLOW", b_shuffle=False):
        '''
        load data, shuffle each feature seperately
        split into training set and test set
        raw_*: 以trail为单位的数据
        *_arr: 以(n_Ep*n_sensors)为单位的数据
        '''
        # self.split_train_test()
        train_idx, test_idx = self.split_train_test(adjective=adjective, b_shuffle=b_shuffle)
        raw_names, raw_adjs_labels = self.get_info_data()

        # infomations are same for different features in each trail
        train_shuffled = self.shuffle_array(raw_names, train_idx)
        train_labels_shuffled = raw_adjs_labels[train_idx, :]
        test_shuffled = self.shuffle_array(raw_names, test_idx)
        test_labels_shuffled = raw_adjs_labels[test_idx, :]

        print("loading training data...")
        train_shuffled = self.get_sensor_data(read_idx=train_idx, explore_period=ep)

        print("loading testing data...")
        test_shuffled = self.get_sensor_data(read_idx=test_idx, explore_period=ep)

        self.train_arr = train_shuffled
        self.test_arr = test_shuffled
        self.train_labels_arr = np.expand_dims(np.array(train_labels_shuffled[:, self.adjectives_mp[adjective]]), axis=1)
        self.test_labels_arr = np.expand_dims(np.array(test_labels_shuffled[:, self.adjectives_mp[adjective]]), axis=1)

    def split_train_test(self, adjective="absorbent", b_shuffle=False):
        idx_train = []
        idx_test = []
        train_file = np.load('./Data/PHAC/split/train_split.npy', allow_pickle=True).item()
        test_file = np.load('./Data/PHAC/split/test_split.npy', allow_pickle=True).item()

        for i in range(self.start_idx, self.end_idx):
            if self.keys[i] in test_file[adjective]:
                idx_test.append(i)
            else:
                idx_train.append(i)

        if b_shuffle:
            np.random.shuffle(idx_train)
            np.random.shuffle(idx_test)

        return idx_train, idx_test

    def shuffle_array(self, arrays_ori, shuffle_idx):
        arrays = []
        for i in shuffle_idx:
            arrays.append(arrays_ori[i])
        return arrays

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
        raw_adjs = np.zeros((self.end_idx - self.start_idx, self.n_adjs), dtype=np.int32)
        for i in range(self.start_idx, self.end_idx):
            k = self.keys[i]
            # raw_labels[i-self.start_idx] = self.keys_mp[k[:-3]] #map the key to the object class
            obj = self.data[k]

            # 形容词
            for a in obj['adjectives']:
                a_str = a.decode('ascii')
                raw_adjs[i - self.start_idx, self.adjectives_mp[a_str]] = 1

        return raw_names, raw_adjs

    def get_sensor_data(self, read_idx, explore_period='SLIDE_5CM'):
        '''
        get a specific kind of data from all objects given sensor name and explore period
        '''
        sensor_data_0 = []
        longest_len = 100
        shortest_len = 2000

        for i in read_idx:
            k = self.keys[i]
            # print(k)

            obj = self.data[k]

            biotac = obj['biotacs']
            finger_0 = biotac['finger_0']
            finger_1 = biotac['finger_1']

            curr_sensor = []
            for sensor in Sensors:
                sensor_data = np.array(finger_0[sensor])
                sensor_data = (sensor_data - np.mean(sensor_data)) / np.std(sensor_data)

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

                # EP_b = np.array(state_encoded) == self.dict_ep[explore_period]
                EP_b = state_arr == explore_period.encode()
                EP_idx = np.argwhere(EP_b)

                # segment sensor data according to EP
                sensor_data_ep = sensor_data[EP_b]  # .reshape(-1)  # sensor_data特征的采样频率是其他的22倍

                longest_len = max(longest_len, sensor_data_ep.shape[0])
                shortest_len = min(shortest_len, sensor_data_ep.shape[0])
                if sensor_data_ep.ndim < 2:
                    sensor_data_ep = np.expand_dims(sensor_data_ep, axis=1)
                curr_sensor.append(sensor_data_ep)

            curr_sensor = np.concatenate(curr_sensor, axis=1)
            sensor_data_0.append(curr_sensor)

        print("Data longest len: " + str(longest_len))
        print("Data shortest len: " + str(shortest_len))
        return sensor_data_0


if __name__ == "__main__":
    handler = File_handler()
    handler.load_data(b_shuffle=True)
    print(handler)