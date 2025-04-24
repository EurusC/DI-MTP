import os
import pickle
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from sklearn.model_selection import train_test_split


def form_train_test(lon_min=121.9, lon_max=122, lat_min=29.93, lat_max=30.05, test_size=0.2, random_state=42, path="./batches.pkl"):
    with open(path, 'rb') as f:
        batches = pickle.load(f)
    traj_batches = []
    mask_batches = []
    mmsi_batches = []

    def normalize(value, min_val, max_val):
        return (value - min_val) / (max_val - min_val)

    for batch in batches:
        traj, mask = batch
        min_length = min(df.shape[0] for df in traj)

        mmsi = [df['mmsi'].iloc[0] for df in traj]

        # 调整traj中的df长度
        traj = [df[['lon', 'lat']].iloc[:min_length] for df in traj]

        # 对每个df的lon和lat列进行归一化
        for df in traj:
            df['lon'] = df['lon'].apply(normalize, args=(lon_min, lon_max))
            df['lat'] = df['lat'].apply(normalize, args=(lat_min, lat_max))
        traj = np.array(traj)
        mask = np.array(mask)
        mmsi = np.array(mmsi)

        traj_batches.append(traj)
        mask_batches.append(mask)
        mmsi_batches.append(mmsi)

    traj_train, traj_test, mask_train, mask_test, mmsi_train, mmsi_test = train_test_split(
        traj_batches, mask_batches, mmsi_batches, test_size=test_size, random_state=random_state)
    return traj_train, traj_test, mask_train, mask_test, mmsi_train, mmsi_test


class DataLoader:
    def __init__(self, file_path, data_scale, lon_min, lon_max, lat_min, lat_max, test_size=0.2, random_state=42, obs_len=6, pred_len=12):
        """
        :param file_path: pickle 文件的路径
        :param lon_min: 经度归一化的最小值
        :param lon_max: 经度归一化的最大值
        :param lat_min: 纬度归一化的最小值
        :param lat_max: 纬度归一化的最大值
        :param test_size: 测试集的比例
        :param random_state: 随机种子
        """
        self.file_path = file_path
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.test_size = test_size
        self.random_state = random_state

        # 载入数据
        traj_train, traj_test, mask_train, mask_test, mmsi_train, mmsi_test = form_train_test(self.lon_min, self.lon_max, self.lat_min,
                                                                       self.lat_max, self.test_size, self.random_state,
                                                                       self.file_path)

        if data_scale > 0:
            for i, _ in enumerate(traj_train):
                traj_train[i] = traj_train[i] * data_scale
            for i, _ in enumerate(traj_test):
                traj_test[i] = traj_test[i] * data_scale
        for i, _ in enumerate(traj_train):
            traj_train[i] = traj_train[i][:, :20]
            traj_train[i] = traj_train[i][:, 20-obs_len-pred_len:]
        for i, _ in enumerate(traj_test):
            traj_test[i] = traj_test[i][:, :20]
            traj_test[i] = traj_test[i][:, 20-obs_len-pred_len:]
        train_init_pos = []
        test_init_pos = []
        for i, _ in enumerate(traj_train):
            train_init_pos.append(traj_train[i][:, obs_len-1:obs_len, ])
            traj_train[i] = traj_train[i] - traj_train[i][:, obs_len-1:obs_len, ]
        for i, _ in enumerate(traj_test):
            test_init_pos.append(traj_test[i][:, obs_len-1:obs_len, ])
            traj_test[i] = traj_test[i] - traj_test[i][:, obs_len-1:obs_len, ]

        self.traj_train = traj_train
        self.traj_test = traj_test
        self.mask_train = mask_train
        self.mask_test = mask_test
        self.train_init_pos = train_init_pos
        self.test_init_pos = test_init_pos
        self.mmsi_train = mmsi_train
        self.mmsi_test = mmsi_test
