import math
import os
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from collections import namedtuple
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter


def set_input_targets(traj, mask, label, route, init_pos, args, device, mmsi=None):
    traj = torch.DoubleTensor(traj).to(device)
    mask = torch.DoubleTensor(mask).to(device)
    label = torch.LongTensor(label).to(device)
    init_pos = torch.DoubleTensor(init_pos).to(device)
    route = torch.DoubleTensor(route).to(device)

    obs_len = args.obs_len
    pred_len = args.pred_len

    obs_traj = traj[:, :obs_len]
    gt_traj = traj[:, obs_len:obs_len + pred_len]

    Inputs = namedtuple('Inputs', ['obs_traj', 'mask', 'init_pos', 'gt_traj', 'label', 'route', 'device', 'tf'])
    inputs = Inputs(obs_traj=obs_traj, mask=mask, init_pos=init_pos, gt_traj=gt_traj, label=label, route=route, device=device, tf=False)

    Targets = namedtuple('Targets', ['label', 'gt_traj', 'init_pos', 'obs', 'mmsi'])
    targets = Targets(label=label, gt_traj=gt_traj, init_pos=init_pos, obs=obs_traj, mmsi=mmsi)

    return inputs, targets


def get_labels(dataloader, kmeans, ae, device):
    train_labels = []
    test_labels = []
    for traj, init_pos in zip(dataloader.traj_train, dataloader.train_init_pos):
        traj = torch.DoubleTensor(traj).to(device)
        init_pos = torch.DoubleTensor(init_pos).to(device)
        enc = ae.encode(traj, init_pos).cpu().detach().numpy().squeeze()
        labels = kmeans.predict(enc)
        train_labels.append(labels)
    for traj, init_pos in zip(dataloader.traj_test, dataloader.test_init_pos):
        traj = torch.DoubleTensor(traj).to(device)
        init_pos = torch.DoubleTensor(init_pos).to(device)
        enc = ae.encode(traj, init_pos).cpu().detach().numpy().squeeze()
        labels = kmeans.predict(enc)
        test_labels.append(labels)
    return train_labels, test_labels


def custom_criterion(outputs, targets):
    mu = outputs['mu']
    log = outputs['log']
    pred_traj = outputs['pred_traj']
    path_score = outputs['path_score']
    pred_route_end = outputs['pred_route_end']
    gt = targets.gt_traj
    gt_end = gt[:, -2:-1, :]

    label = targets.label

    pred_traj = pred_traj.reshape(gt.shape)

    future_loss = F.smooth_l1_loss(pred_traj, gt)

    pred_route_end = pred_route_end.reshape(gt_end.shape)

    route_end_loss = F.smooth_l1_loss(pred_route_end, gt_end)
    clf_loss = F.cross_entropy(path_score, label)
    kld = -0.5 * torch.sum(1 + log - mu.pow(2) - log.exp())
    kld = kld / gt.shape[0]
    loss = future_loss + route_end_loss + clf_loss + kld * 0.01

    return {'loss': loss, 'end_loss': route_end_loss, 'future_loss': future_loss, 'clf_loss': clf_loss, 'kld': kld}


def custom_metrics(outputs, targets, args):
    pred_traj = outputs['pred_traj']
    prob = outputs['prob']
    obs = targets.obs
    gt = targets.gt_traj
    init_pos = targets.init_pos
    pred = pred_traj.reshape((gt.shape[0], args.num_k, gt.shape[1], gt.shape[2]))
    gt = gt + init_pos
    init_pos = init_pos.unsqueeze(1)
    pred = pred + init_pos
    obs = obs + init_pos.squeeze(1)
    gt = gt.unsqueeze(1)
    norm_ = torch.norm(pred - gt, p=2, dim=-1)
    ade_ = torch.mean(norm_, dim=-1)
    fde_ = norm_[:, :, -1]
    min_ade, _ = torch.min(ade_, dim=-1)
    min_fde, _ = torch.min(fde_, dim=-1)
    min_ade = torch.sum(min_ade) / pred_traj.shape[0]
    min_fde = torch.sum(min_fde) / pred_traj.shape[0]
    # obs[:, :, 0] = denormalize(obs[:, :, 0], args.lon_min, args.lon_max)
    # obs[:, :, 1] = denormalize(obs[:, :, 1], args.lat_min, args.lat_max)
    # pred[:, :, :, 0] = denormalize(pred[:, :, :, 0], args.lon_min, args.lon_max)
    # pred[:, :, :, 1] = denormalize(pred[:, :, :, 1], args.lat_min, args.lat_max)
    # gt[:, :, :, 0] = denormalize(gt[:, :, :, 0], args.lon_min, args.lon_max)
    # gt[:, :, :, 1] = denormalize(gt[:, :, :, 1], args.lat_min, args.lat_max)
    # mmsi = targets.mmsi
    # plot_time_steps(pred, gt, obs, args.batch_num, mmsi)
    # all_x = torch.concat([gt.squeeze()[:, :, 0].flatten(), obs[:, :, 0].flatten(), pred[:, :, :, 0].flatten()])
    # all_y = torch.concat([gt.squeeze()[:, :, 1].flatten(), obs[:, :, 1].flatten(), pred[:, :, :, 1].flatten()])
    # x_min, x_max = torch.min(all_x), torch.max(all_x)
    # y_min, y_max = torch.min(all_y), torch.max(all_y)
    # plot_predictions_vs_ground_truth(pred, gt, obs, args.epoch, args.batch_num, mmsi=mmsi, prob=prob)
    return {'min_ade': min_ade, 'min_fde': min_fde}

    # 设置 x 轴数字格式化为保留 1 位小数


def format_x(value, tick_number):
    return f"{value:.1f}"

    # 设置 y 轴数字格式化为保留 2 位小数


def format_y(value, tick_number):
    return f"{value:.2f}"


def get_ade_fde_true(outputs, targets, args, outputdir):
    pred_traj = outputs['pred_traj']
    obs = targets.obs
    gt = targets.gt_traj
    init_pos = targets.init_pos
    num_k = args.num_k
    pred_trajs = pred_traj.reshape(gt.shape[0], num_k, gt.shape[1], -1)
    gt = gt + init_pos
    obs = obs + init_pos
    init_pos = init_pos.unsqueeze(1)
    pred_trajs = pred_trajs + init_pos
    ade = np.zeros((pred_trajs.shape[0], num_k))
    fde = np.zeros((pred_trajs.shape[0], num_k))
    for i in range(num_k):
        pre_trajs = pred_trajs[:, i]
        for j in range(pre_trajs.shape[0]):
            p_traj = pre_trajs[j:j + 1]
            gt_traj = gt[j:j + 1]
            ade_, fde_ = calculate_ade_fde(p_traj, gt_traj, args)
            ade[j, i] = ade_
            fde[j, i] = fde_
    ade_fde_sum = ade + fde
    min_indices = np.argmin(ade_fde_sum, axis=-1)  # 找到 ade + fde 最小值的索引

    min_ade = ade[np.arange(ade.shape[0]), min_indices]
    min_fde = fde[np.arange(fde.shape[0]), min_indices]

    min_ade = torch.from_numpy(min_ade)
    min_fde = torch.from_numpy(min_fde)

    min_ade = torch.sum(min_ade) / pred_trajs.shape[0]
    min_fde = torch.sum(min_fde) / pred_trajs.shape[0]

    # for i in range(pred_trajs.shape[0]):
    #     pred_trajs[i, :, :, 0] = denormalize(pred_trajs[i, :, :, 0] / args.data_scale, args.lon_min, args.lon_max)
    #     pred_trajs[i, :, :, 1] = denormalize(pred_trajs[i, :, :, 1] / args.data_scale, args.lat_min, args.lat_max)
    # gt[:, :, 0] = denormalize(gt[:, :, 0] / args.data_scale, args.lon_min, args.lon_max)
    # gt[:, :, 1] = denormalize(gt[:, :, 1] / args.data_scale, args.lat_min, args.lat_max)
    #
    # obs[:, :, 0] = denormalize(obs[:, :, 0] / args.data_scale, args.lon_min, args.lon_max)
    # obs[:, :, 1] = denormalize(obs[:, :, 1] / args.data_scale, args.lat_min, args.lat_max)
    # prob = outputs['prob']
    # mmsi = targets.mmsi
    # plot_time_steps(pred_trajs, gt, obs, args.batch_num, mmsi)
    # all_x = torch.concat([gt.squeeze()[:, :, 0].flatten(), obs[:, :, 0].flatten(), pred_trajs[:, :, :, 0].flatten()])
    # all_y = torch.concat([gt.squeeze()[:, :, 1].flatten(), obs[:, :, 1].flatten(), pred_trajs[:, :, :, 1].flatten()])
    # x_min, x_max = torch.min(all_x), torch.max(all_x)
    # y_min, y_max = torch.min(all_y), torch.max(all_y)
    # plot_predictions_vs_ground_truth(pred_trajs, gt, obs, args.epoch, args.batch_num, mmsi=mmsi, prob=prob, outputdir=outputdir)
    return {'min_ade': min_ade, 'min_fde': min_fde}


def plot_predictions_vs_ground_truth(pred, gt, obs, epoch, batch_num, mmsi=None, x_min=None, x_max=None, y_min=None, y_max=None, prob=None, outputdir='./vis_plots'):
    pred_np = pred.detach().cpu().numpy()   # (batch_size, num_samples, num_timesteps, 2)
    gt_np = gt.detach().cpu().numpy()       # (batch_size, num_timesteps, 2)
    if len(gt_np.shape) == 4:
        gt_np = gt_np.squeeze(1)
    obs_np = obs.detach().cpu().numpy()     # (batch_size, obs_timesteps, 2)
    mmsi = mmsi
    is_prob = False
    if prob is not None:
        prob = prob.detach().cpu().numpy()  # (batch_size, num_samples)
        is_prob = True
    gt_marker_size = 8
    pred_marker_size = 5
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    pred_markers = ['s', '^', 'p', 'D', 'v', 'h', 'x', '*', 'o', '+']

    plt.figure(figsize=(8, 6))
    for i in range(gt_np.shape[0]):  # 遍历每个 batch_size 中的轨迹
        color = colors[i]

        plt.plot(
            gt_np[i, :, 0], gt_np[i, :, 1], color='black', linestyle='-', marker='o',
            linewidth=2, markersize=gt_marker_size, label=f'Ground Truth' if i == 0 else ""
        )

        for j in range(pred_np.shape[1]):  # 遍历每条轨迹的 num_samples 个预测样本
            plt.plot(
                pred_np[i, j, :, 0], pred_np[i, j, :, 1], color=color, linestyle='--',
                marker=pred_markers[i], linewidth=1, markersize=pred_marker_size,
                label=f'Prediction (MMSI {mmsi[i]})' if j == 0 else ""
            )

            if is_prob:
                x_last, y_last = pred_np[i, j, -1, 0], pred_np[i, j, -1, 1]  # 获取预测轨迹的最后一个点

                # 设置偏移量，避免标注和轨迹点重叠
                x_offset = 0.001  # 根据坐标范围调整
                y_offset = 0.001

                # 添加标注，位置偏移
                plt.text(
                    x_last + x_offset, y_last + y_offset, f"{prob[i, j]:.4f}",
                    color=color, fontsize=10, ha='left', va='bottom'
                )

    for i in range(obs_np.shape[0]):  # batch_size

        plt.plot(
            obs_np[i, :, 0], obs_np[i, :, 1], color='#FFA500', linestyle='-', marker='o',
            linewidth=2, markersize=gt_marker_size, label=f'Observed' if i == 0 else '')

        plt.plot([obs_np[i, -1, 0], gt_np[i, 0, 0]], [obs_np[i, -1, 1], gt_np[i, 0, 1]],
                 color='#FFA500', linestyle='-', linewidth=2)

    plt.xlabel("Longitude", fontsize=18)
    plt.ylabel("Latitude", fontsize=18)

    # 添加图例，避免重复显示，调整位置避免遮挡
    plt.legend(loc="upper right", fontsize=16)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    ax.tick_params(axis='x', labelsize=16)  # 设置 x 轴刻度标签字体大小
    ax.tick_params(axis='y', labelsize=16)  # 设置 y 轴刻度标签字体大小

    path = outputdir + f"/{epoch + 1}/"
    if not os.path.exists(path):
        os.makedirs(path)
    title = '_'.join(map(str, mmsi)) + f"_{batch_num}"
    plt.savefig(path + f"{title}.png", bbox_inches='tight', dpi=300)
    plt.close()


def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000  # Radius of Earth in meters
    return c * r


def calculate_ade_fde(pred_traj, gt_traj, args):
    batch_size, time_steps, _ = pred_traj.shape
    lon_max = args.lon_max
    lat_max = args.lat_max
    lat_min = args.lat_min
    lon_min = args.lon_min
    data_scale = args.data_scale
    ade = 0
    fde = 0
    for i in range(batch_size):
        traj_ade = 0
        for t in range(time_steps):
            pred_lon = denormalize(pred_traj[i, t, 0] / data_scale, lon_min, lon_max)
            pred_lat = denormalize(pred_traj[i, t, 1] / data_scale, lat_min, lat_max)
            gt_lon = denormalize(gt_traj[i, t, 0] / data_scale, lon_min, lon_max)
            gt_lat = denormalize(gt_traj[i, t, 1] / data_scale, lat_min, lat_max)
            distance = haversine(pred_lon, pred_lat, gt_lon, gt_lat)
            traj_ade += distance
        traj_ade /= time_steps
        ade += traj_ade
        pred_lon = denormalize(pred_traj[i, -1, 0] / data_scale, lon_min, lon_max)
        pred_lat = denormalize(pred_traj[i, -1, 1] / data_scale, lat_min, lat_max)
        gt_lon = denormalize(gt_traj[i, -1, 0] / data_scale, lon_min, lon_max)
        gt_lat = denormalize(gt_traj[i, -1, 1] / data_scale, lat_min, lat_max)
        fde += haversine(pred_lon, pred_lat, gt_lon, gt_lat)
    ade /= batch_size
    fde /= batch_size
    return ade, fde


def denormalize(value, min_value, max_value):
    return value * (max_value - min_value) + min_value


def num_k_ade_fde_long(param_config, model_config, model, test_func, dataloader, device, pic_name):
    log_name = pic_name.split('ade')[0] + 'long.txt'
    ade_values = []  # 用于存储 ade 值
    fde_values = []  # 用于存储 fde 值
    num_k_values = range(1, param_config.n_clusters + 1)

    with open(log_name, 'w') as log_file:
        for i in num_k_values:
            param_config.route_num = i
            param_config.num_k = param_config.route_num
            metrics_dict = test_func(model, dataloader, 100000, get_ade_fde_true, device, model_config, param_config)
            ade_values.append(metrics_dict['min_ade'].item())
            fde_values.append(metrics_dict['min_fde'].item())
            # 将输出写入到文件中
            log_file.write(f"num_k [{i}] ade: [{metrics_dict['min_ade']}] fde: [{metrics_dict['min_fde']}]\n")

    sns.set(style="whitegrid", palette="muted", font_scale=1.5)
    data = {
        "num_k": list(num_k_values) * 2,  # 横轴 num_k 需要重复两次
        "Metric Value": ade_values + fde_values,  # 纵轴的 ade 和 fde 值
        "Metric Type": ["ADE"] * len(ade_values) + ["FDE"] * len(fde_values)  # 区分ADE和FDE
    }

    df = pd.DataFrame(data)

    # 设置绘图大小
    plt.figure(figsize=(10, 6))

    sns.lineplot(x="num_k", y="Metric Value", hue="Metric Type", style="Metric Type", markers=True, dashes=False,
                 data=df)

    # 设置标题和标签
    plt.title('ADE and FDE vs num_k', fontsize=18)
    plt.xlabel('num_k', fontsize=14)
    plt.ylabel('ADE / FDE', fontsize=14)

    # 设置横轴的刻度为整数
    plt.xticks(np.arange(min(num_k_values), max(num_k_values) + 1, 1), fontsize=8)

    plt.grid(False)

    # 保存图片到文件
    plt.savefig(pic_name, bbox_inches='tight', dpi=300)  # bbox_inches避免标题被截断，dpi提高分辨率


def num_k_ade_fde_short(param_config, model_config, model, test_func, dataloader, device, pic_name):
    log_name = pic_name.split('ade')[0] + 'short.txt'
    ade_values = []  # 用于存储 ade 值
    fde_values = []  # 用于存储 fde 值

    num_k_values = range(1, param_config.num_k + 1)
    param_config.route_num = 1
    with open(log_name, 'w') as log_file:
        for i in num_k_values:
            param_config.num_k = i
            metrics_dict = test_func(model, dataloader, 100000, get_ade_fde_true, device, model_config, param_config)
            ade_values.append(metrics_dict['min_ade'].item())
            fde_values.append(metrics_dict['min_fde'].item())
            # 将输出写入到文件中
            log_file.write(f"num_k [{i}] ade: [{metrics_dict['min_ade']}] fde: [{metrics_dict['min_fde']}]\n")

    sns.set(style="whitegrid", palette="muted", font_scale=1.5)
    data = {
        "num_k": list(num_k_values) * 2,  # 横轴 num_k 需要重复两次
        "Metric Value": ade_values + fde_values,  # 纵轴的 ade 和 fde 值
        "Metric Type": ["ADE"] * len(ade_values) + ["FDE"] * len(fde_values)  # 区分ADE和FDE
    }

    df = pd.DataFrame(data)

    # 设置绘图大小
    plt.figure(figsize=(10, 6))

    sns.lineplot(x="num_k", y="Metric Value", hue="Metric Type", style="Metric Type", markers=True, dashes=False,
                 data=df)

    plt.title('ADE and FDE vs num_k', fontsize=18)
    plt.xlabel('num_k', fontsize=14)
    plt.ylabel('ADE / FDE', fontsize=14)

    plt.xticks(np.arange(min(num_k_values), max(num_k_values) + 1, 1), fontsize=8)

    plt.grid(False)

    plt.savefig(pic_name, bbox_inches='tight', dpi=300)  # bbox_inches避免标题被截断，dpi提高分辨率


def plot_time_steps(pred, gt, obs, batch_num, mmsi=None):
    pred_np = pred.detach().cpu().numpy()
    obs_np = obs.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    if len(gt_np.shape) == 4:
        gt = gt_np.squeeze(1)
    else:
        gt = gt_np
    title = ''
    for msi in mmsi:
        title += str(msi)
        title += '_'
    title = title + str(batch_num)

    num_timesteps = pred_np.shape[2]  # 预测的时间步数

    dist = np.linalg.norm(pred_np - np.expand_dims(gt, axis=1), axis=(2, 3))
    dist = np.argmin(dist, axis=1)  # 选择距离最小的预测
    pred_np = pred_np[np.arange(pred_np.shape[0]), dist]  # 获取最接近真实轨迹的预测

    ncols = 6
    nrows = math.ceil(num_timesteps / ncols)  # 动态计算行数

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 10))  # 更大的画布大小
    axes = axes.flatten()

    for t in range(num_timesteps):
        ax = axes[t]
        ax.set_title(f"Timestep {t + 1}", fontsize=12)

        for i in range(obs_np.shape[0]):
            ax.plot(obs_np[i, :, 0], obs_np[i, :, 1], color='#FFD700', linestyle='-', marker='o', linewidth=2,
                    markersize=4, alpha=0.8,
                    label='Observed' if i == 0 else "")

            ax.scatter(pred_np[i, t, 0], pred_np[i, t, 1], marker='p', color='r', s=100,  # 使用五角星
                       label='Current Prediction'.format(t + 1) if i == 0 else "")

            if t > 0:
                ax.scatter(pred_np[i, :t, 0], pred_np[i, :t, 1], marker='o', color='r', s=10,
                           label='Prediction' if i == 0 else "")

        ax.xaxis.set_major_formatter(FuncFormatter(format_x))  # x 轴保留 1 位小数
        ax.yaxis.set_major_formatter(FuncFormatter(format_y))  # y 轴保留 2 位小数

        ax.grid(False)
        if t == 0:
            ax.legend(loc="best", fontsize=10)

    for extra_ax in axes[num_timesteps:]:
        fig.delaxes(extra_ax)

    path = f"./timesteps/"
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + f"{title}.png", bbox_inches='tight', dpi=300)
    plt.close()


def dataset_vis(dataloader, args):
    traj_train = dataloader.traj_train
    train_init_pos = dataloader.train_init_pos
    traj_test = dataloader.traj_test
    test_init_pos = dataloader.test_init_pos
    trajs = []

    for i in range(len(traj_train)):
        traj_train[i] = traj_train[i] + train_init_pos[i]
        for j in range(len(traj_train[i])):
            trajs.append(traj_train[i][j])

    for i in range(len(traj_test)):
        traj_test[i] = traj_test[i] + test_init_pos[i]
        for j in range(len(traj_test[i])):
            trajs.append(traj_test[i][j])

    trajs = np.array(trajs)
    if trajs.ndim != 3 or trajs.shape[2] != 2:
        raise ValueError("Trajectory data must be a (n, m, 2) shaped array.")

    trajs[:, :, 0] = denormalize(trajs[:, :, 0], args.lon_min, args.lon_max)  # 经度
    trajs[:, :, 1] = denormalize(trajs[:, :, 1], args.lat_min, args.lat_max)  # 纬度

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='gray')
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='white')  # 使用浅色海洋背景
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')

    gl = ax.gridlines(draw_labels=True, color='none', alpha=0, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False

    min_lon, max_lon = np.min(trajs[:, :, 0]), np.max(trajs[:, :, 0])
    min_lat, max_lat = np.min(trajs[:, :, 1]), np.max(trajs[:, :, 1])
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    colors = plt.cm.Greys(np.linspace(0.4, 0.8, len(trajs)))  # 使用 'Blues' 颜色映射，稍微偏鲜艳
    for i in range(len(trajs)):
        ax.plot(trajs[i, :, 0], trajs[i, :, 1], color=colors[i], lw=1.5, alpha=0.7)  # 稍微增加透明度
        # ax.plot(trajs[i, :, 0], trajs[i, :, 1], color='gray', lw=1.5, alpha=0.7)  # 稍微增加透明度

    ax.set_title('Trajectory Visualization with Map Background', fontsize=10, weight='bold')

    plt.savefig('trajectory_visualization_paper_style.png', bbox_inches='tight', dpi=300)

    # 显示图表
    plt.show()



