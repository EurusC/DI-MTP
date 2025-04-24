import glob
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data_utils import (lon_lat_filter, get_mmsi_trajs, segment_trajectory_df, sample_traj, cal_trajs_impacts,
                        extract_social_data, plot_encounters, plot_traj_, address_time, find_most_congested_time_range, LATITUDE_FORMATTER, LONGITUDE_FORMATTER, ccrs, vis_batches)

def load_csv(src):
    if 'hainan' in src:
        lon_min = 109
        lon_max = 109.22
        lat_min = 18.3
        lat_max = 18.4
    elif 'zhoushan' in src:
        lon_min = 122.2
        lon_max = 122.8
        lat_min = 30.2
        lat_max = 30.7
    elif 'mask' in src:
        lon_min = 92.215
        lon_max = 92.27498666666666
        lat_min = 0.20000333333333
        lat_max = 0.24997
    files = glob.glob(src + "*.csv")
    batches = []
    for file in files:
        df = pd.read_csv(file)
        if 'hainan' in src:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            start_date = df['timestamp'].min()
            end_date = start_date + pd.Timedelta(days=1)
            df = df[(df['timestamp'] >= start_date) & (df['timestamp'] < end_date)]
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9            
        df.columns = ['timestamp', 'MMSI', 'Lon', 'Lat', 'Cog', 'Sog']
        trajs, names = get_mmsi_trajs(df)
        tmp = []
        for traj in trajs:
            traj = lon_lat_filter(traj, lon_min, lon_max, lat_min, lat_max)
            if len(traj) > 0:
                tmp.append(traj)
        trajs = tmp
        segments = []
        for traj in trajs:
            segments.extend(segment_trajectory_df(traj, 600))
        sampled_trajs = []
        for traj in segments:
            if len(traj) < 10:
                continue
            sampled_traj = sample_traj(traj, 30)
            sampled_traj['mmsi'] = traj['MMSI'].values.tolist()[0]
            sampled_trajs.append(sampled_traj)
        count = -1
        for i, traj in enumerate(sampled_trajs):
            target_traj = traj
            compare_trajs = sampled_trajs[:i] + sampled_trajs[i+1:]
            time_range_data, time_range = address_time(target_traj, compare_trajs)
            compare_trajs.append(target_traj)
            index, res = find_most_congested_time_range(compare_trajs, time_range, 10)
            if index is None:
                continue
            count += 1
            batch = res
            tcpa_range = [-0.3, 0.8]
            dcpa_range = [0, 2]
            impacts = cal_trajs_impacts(batch, tcpa_range, dcpa_range)
            batches.append((batch, impacts))
            # plot_encounters(batch, f"./encounter_plots/{count}_")
    print("找到了" + str(len(batches)) + "个会遇场景")
    with open('./data/mask_batches.pkl', 'wb') as file:
        pickle.dump(batches, file)


def get_batches_range():
    path = './s_batches.pkl'
    with open(path, 'rb') as f:
        batches = pickle.load(f)
    lon_max = 0
    lon_min = 180
    lat_max = 0
    lat_min = 90
    for batch in batches:
        traj, mask = batch
        for df in traj:
            if df['lon'].max() > lon_max:
                lon_max = df['lon'].max()
            if df['lon'].min() < lon_min:
                lon_min = df['lon'].min()
            if df['lat'].max() > lat_max:
                lat_max = df['lat'].max()
            if df['lat'].min() < lat_min:
                lat_min = df['lat'].min()
    print(lon_min, lon_max, lat_min, lat_max)


if __name__ == '__main__':
    src = "./orig_data/mask/"
    load_csv(src)
