import glob
import os.path
import datetime
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from tqdm import tqdm
import pickle
from encounter import judge_impact
import cartopy.crs as ccrs
import matplotlib.cm as cm
from matplotlib import pyplot as plt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def lon_lat_filter(data, lon_min=121.9, lon_max=122, lat_min=29.93, lat_max=30.05, min_sog=5, max_sog=30):
    filtered_data = data[
        (data['Lon'] >= lon_min) & (data['Lon'] <= lon_max) &
        (data['Lat'] >= lat_min) & (data['Lat'] <= lat_max) &
        (data['Sog'] >= min_sog) & (data['Sog'] <= max_sog)
    ]
    return filtered_data

def get_mmsi_trajs(df):
    mmsi = df.groupby('MMSI')
    trajs = []
    names = []
    for msi, d in mmsi:
        d = d.sort_values(by=['timestamp'], ascending=True)
        d = d.drop_duplicates(subset=['timestamp', 'MMSI'])
        # d.set_index('timestamp', inplace=True)
        trajs.append(d)
        names.append(msi)
    return trajs, names


def segment_trajectory_df(df, time_threshold):
    segments = []
    current_segment = [df.iloc[0]]

    for i in range(1, len(df)):
        time_diff = df.iloc[i]['timestamp'] - df.iloc[i - 1]['timestamp']

        if time_diff > time_threshold:
            segments.append(pd.concat(current_segment, axis=1).T)
            current_segment = [df.iloc[i]]
        else:
            current_segment.append(df.iloc[i])

    if current_segment:
        segments.append(pd.concat(current_segment, axis=1).T)

    return segments


def sample_traj(traj, time_interval):
    # 将时间戳转换为DatetimeIndex
    traj['timestamp'] = pd.to_datetime(traj['timestamp'], unit='s')
    traj.set_index('timestamp', inplace=True)

    # 重新采样，根据time_interval创建新的时间序列，并进行插值
    sampled_segment = traj.resample(f'{time_interval}S').mean().interpolate(method='time')

    # 将时间索引重置为普通列
    sampled_segment.reset_index(inplace=True)
    sampled_segment.rename(columns={'MMSI': 'mmsi', 'Lon': 'lon', 'Lat': 'lat', 'Cog': 'cog', 'Sog': 'sog'}, inplace=True)

    return sampled_segment



def cal_trajs_impacts(trajs, tcpa_range, dcpa_range):
    impacts = np.zeros(shape=(len(trajs), len(trajs)))
    for i in range(len(trajs)):
        for j in range(len(trajs)):
            if i == j:
                continue
            if trajs[i]['mmsi'].iloc[0] == trajs[j]['mmsi'].iloc[0]:
                continue
            if judge_impact(trajs[i], trajs[j], tcpa_range, dcpa_range):
                impacts[i][j] = 1
                impacts[j][i] = 1
    return impacts


def extract_social_data(trajs, impacts):
    res = []
    for i in range(len(trajs)):
        relations = []
        for j in range(len(trajs)):
            if i == j:
                relations.append(trajs[j])
                continue
            if trajs[i]['mmsi'].iloc[0] == trajs[j]['mmsi'].iloc[0]:
                continue
            if impacts[i][j] == 1:
                relations.append(trajs[j])
        if len(relations) == 1:
            continue
        res.append(relations)
    return res


def plot_encounters(trajs, save_path="./encounter_plots/"):
    last_slash_index = save_path.rfind('/')

    # 提取最后一个斜杠之前的所有字符串
    if last_slash_index != -1:
        result = save_path[:last_slash_index] + "/"
    else:
        result = save_path
    if not os.path.exists(result):
        os.makedirs(result)
    colors = cm.viridis(np.linspace(0, 10, len(trajs)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # ax.set_extent([121.9, 122, 29.93, 30.05])  # 进一步缩小地图显示范围和实现更大的缩放级别
    ax.coastlines()
    ax.stock_img()
    count = 0
    for i in tqdm(range(len(trajs))):
        plot_traj(trajs[i], ax, colors[count])
    file_name = save_path + str(len(trajs)) + "_encounter" + str('.png')
    plt.savefig(file_name)
    plt.close()


def plot_traj(df, ax, c='b'):
    ax.scatter(df['lon'], df['lat'], transform=ccrs.PlateCarree(), s=5, color=c)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER


def plot_traj_(df, c='b', save="./traj_plots/"):
    if not os.path.exists(save):
        os.makedirs(save)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([121.9, 122, 29.93, 30.05])  # 进一步缩小地图显示范围和实现更大的缩放级别
    ax.coastlines()
    ax.stock_img()

    ax.scatter(df['lon'], df['lat'], transform=ccrs.PlateCarree(), s=5, color=c)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    file_name = save + str(df['mmsi'].iloc[0]) + str('.png')
    plt.savefig(file_name)
    plt.close()


def address_time(traj, trajs):
    target_time_min = traj['timestamp'].min()
    target_time_max = traj['timestamp'].max()
    time_range_data = []
    time_range = []

    for tra in trajs:
        tra_time_min = tra['timestamp'].min()
        tra_time_max = tra['timestamp'].max()

        if tra_time_max >= target_time_min and tra_time_min <= target_time_max:
            start = max(tra_time_min, target_time_min)
            end = min(tra_time_max, target_time_max)

            data = (
                traj[(traj['timestamp'] >= start) & (traj['timestamp'] <= end)],
                tra[(tra['timestamp'] >= start) & (tra['timestamp'] <= end)]
            )
            time_range_data.append(data)
            time_range.append((start, end))

    return time_range_data, time_range


def find_most_congested_time_range(trajs, time_range, min_length=10):
    index = -1  # 初始化为-1，表示没有找到符合条件的时间段
    curr_count = 0
    candidates = []  # 存储符合条件的候选索引

    for i, time_r in enumerate(time_range):
        count = 0
        start, end = time_r

        # 检查时间范围是否满足最小长度要求
        if end - start < datetime.timedelta(minutes=min_length):
            continue

        # 统计在这个时间范围内的轨迹数量
        for traj in trajs:
            traj_time_min = traj['timestamp'].min()
            traj_time_max = traj['timestamp'].max()

            # 判断轨迹是否完全在start和end之间
            if start >= traj_time_min and traj_time_max >= end:
                count += 1

        # 如果找到了更大的count，更新候选列表
        if count > curr_count:
            curr_count = count
            candidates = [i]  # 更新候选列表，保留当前最大count的index
        elif count == curr_count:
            candidates.append(i)  # 如果count相等，添加到候选列表

    # 从候选列表中找出第一个满足时间长度要求的index
    for i in candidates:
        start, end = time_range[i]
        if end - start >= datetime.timedelta(minutes=min_length):
            index = i
            break

    # 最后检查是否找到了符合条件的时间段
    if index != -1:
        start, end = time_range[index]
    else:
        # 处理没有找到符合条件的情况
        return None, None
    res = []
    for traj in trajs:
        traj_time_min = traj['timestamp'].min()
        traj_time_max = traj['timestamp'].max()
        if start >= traj_time_min and traj_time_max >= end:
            res.append(traj[(traj['timestamp'] >= start) & (traj['timestamp'] <= end)])
    return index, res


def find_range_by_human():
    src = "D:\\ais dataset\\decoded_ningbo_zhoushan_2022\\"
    files = glob.glob(src + "*.csv")
    files = files[:1]
    batches = []
    lon_min = 122.215
    lon_max = 122.275
    lat_min = 30.2
    lat_max = 30.25
    for file in files:
        df = pd.read_csv(file)
        df = lon_lat_filter(df, lon_min, lon_max, lat_min, lat_max)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_extent([lon_min, lon_max, lat_min, lat_max])  # 设置地图显示范围
        ax.coastlines()
        ax.stock_img()
        ax.scatter(df['Lon'], df['Lat'], transform=ccrs.PlateCarree(), s=1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                          linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        plt.show()


def vis_batches(lon_min, lon_max, lat_min, lat_max):
    src = "s_batches.pkl"

    with open(src, 'rb') as f:
        batches = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max])  # 设置地图显示范围
    ax.coastlines()
    ax.stock_img()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5,
                      linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    for batch in batches:
        traj, mask = batch
        for df in traj:
            ax.scatter(df['lon'], df['lat'], transform=ccrs.PlateCarree(), s=1)
    fig.show()
