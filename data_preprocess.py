import glob
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from data_utils import (lon_lat_filter, get_mmsi_trajs, segment_trajectory_df, sample_traj, cal_trajs_impacts,
                        extract_social_data, plot_encounters, plot_traj_, address_time, find_most_congested_time_range, LATITUDE_FORMATTER, LONGITUDE_FORMATTER, ccrs, vis_batches)

'''
    1. 在lon_lat过滤前，对数据整体处理
    2. MMSI、分段、指定时间范围、采样插值
    3. 对得到的整体数据按照lon_lat过滤
    理论上来说很美好，但是实现过程可能有如下问题：
    1. 采样插值过程中可能出现部分MMSI仅出现几次，这种不能支持全部时间数据的插值
    2. 可以插值，但是仍然有不能插值的部分
'''


def load_csv(src):
    if 'hainan' in src:
        lon_min = 109
        lon_max = 109.22
        lat_min = 18.3
        lat_max = 18.4
    elif 'zhoushan' in src:
        lon_min = 122.215
        lon_max = 122.275
        lat_min = 30.2
        lat_max = 30.25

    files = glob.glob(src + "*.csv")
    files = files[:60]
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
    print(len(batches))
    with open('hainan_batches.pkl', 'wb') as file:
        pickle.dump(batches, file)


def batches_split_2_batch(lon_min, lon_max, lat_min, lat_max):
    src = "s_batches.pkl"
    with open(src, 'rb') as f:
        batches = pickle.load(f)
    for i, batch in enumerate(batches):
        traj, mask = batch
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])  # 设置地图显示范围
        ax.coastlines()
        ax.stock_img()
        for df in traj:
            ax.plot(df['lon'], df['lat'], transform=ccrs.PlateCarree())
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        fig_path = "./batches_splits/plot/"
        data_path = "./batches_splits/data/"
        if not os.path.exists(fig_path):
            os.makedirs(fig_path)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        fig.savefig(fig_path + f"{i}.png")
        with open(data_path + f"{i}.pickle", 'wb') as f:
            pickle.dump(batch, f)


def select_batch_by_pic():
    batch_path = "./batches_splits/data/"
    fig_path = "./batches_splits/plot/"
    pics = glob.glob(fig_path + "*.png")
    indices = [pic.split('\\')[-1].split('.')[0] + ".pickle" for pic in pics]
    batches = []
    for index in indices:
        with open(batch_path + index, 'rb') as f:
            batch = pickle.load(f)
        batches.append(batch)
    save_path = './s_batches.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(batches, f)


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
    '''
        下列为宁波港轨迹经纬度限制
    '''
    src = "./orig_data/"
    hai_nan_src = "./hainan/"

    load_csv(hai_nan_src)
