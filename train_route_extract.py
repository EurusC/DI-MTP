import numpy as np
import matplotlib.pyplot as plt
from PIL.ImageOps import exif_transpose
from sklearn.cluster import KMeans
from model.model import AE
from config import Config
import torch
import torch.nn.functional as F
from dataset import DataLoader
from torch.optim import Adam, lr_scheduler
import os


def custom_criterion(outputs, targets):
    pred_traj = outputs['rec']
    pred_traj = pred_traj.reshape(targets.shape)
    loss = F.smooth_l1_loss(pred_traj, targets)
    return {'loss': loss}


def train(model, optimizer, traj_train, train_init, criterion, device):
    model.train()  # 设置模型为训练模式
    loss_dict = {}
    traj_train = traj_train.to(device)
    train_init = train_init.to(device)

    # 清零优化器的梯度
    optimizer.zero_grad()

    outputs = model(traj_train, train_init)
    targets = traj_train

    res_dict = criterion(outputs, targets)

    for key, value in res_dict.items():
        if key not in loss_dict:
            loss_dict[key] = 0  # 初始化为0 或其他适当的初始值
        loss_dict[key] += value.item()

    loss = res_dict['loss']
    loss.backward()
    optimizer.step()

    for key, value in loss_dict.items():
        loss_dict[key] = value / 1
    return loss_dict


def test(model, traj_test, test_init, criterion, device):
    model.eval()  # 设置模型为评估模式
    metrics_dict = {}  # 初始化 metrics_dict

    with torch.no_grad():  # 禁用梯度计算
        traj = traj_test.to(device)
        test_init = test_init.to(device)

        outputs = model(traj, test_init)

        res_dict = criterion(outputs, traj)
        for key, value in res_dict.items():
            if key not in metrics_dict:
                metrics_dict[key] = 0  # 初始化为0 或其他适当的初始值
            metrics_dict[key] += value.item()

    for key, value in metrics_dict.items():
        metrics_dict[key] = value / 1

    return metrics_dict


def train_ae(model_config, param_config, device):
    seed = param_config.random_state
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 2. 加载数据集
    dataloader = DataLoader(param_config.batches_path, param_config.data_scale, param_config.lon_min,
                            param_config.lon_max, param_config.lat_min, param_config.lat_max, param_config.test_size,
                            param_config.random_state, model_config.obs_len, model_config.pred_len)
    traj_train = []
    traj_len = model_config.obs_len + model_config.pred_len

    for traj in dataloader.traj_train:
        for tra in traj:
            traj_train.append(tra[:traj_len])
    traj_train = np.array(traj_train)
    traj_train = torch.DoubleTensor(traj_train)

    traj_test = []
    for traj in dataloader.traj_test:
        for tra in traj:
            traj_test.append(tra[:traj_len])
    traj_test = np.array(traj_test)
    traj_test = torch.DoubleTensor(traj_test)

    init_train = []
    for init_pos in dataloader.train_init_pos:
        for init in init_pos:
            init_train.append(init)
    init_train = np.array(init_train)
    init_train = torch.DoubleTensor(init_train)

    init_test = []
    for init_pos in dataloader.test_init_pos:
        for init in init_pos:
            init_test.append(init)
    init_test = np.array(init_test)
    init_test = torch.DoubleTensor(init_test)

    epochs = 1000

    # 3. 创建model
    model = AE(model_config).to(device)
    model.double()
    optimizer = Adam(lr=param_config.learning_rate, params=model.parameters())
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[400, 600, 800], gamma=param_config.gama)

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        loss_dict = train(model, optimizer, traj_train, init_train, custom_criterion, device)

        loss_str = ""
        for key in loss_dict:
            loss_str += f"{key} = {loss_dict[key]}   "

        error_dict = test(model, traj_test, init_test, custom_criterion, device)
        error_str = ""
        for key in error_dict:
            error_str += f"{key} = {error_dict[key]}   "
        print(f"\r[Epoch {epoch + 1}] {loss_str} {error_str}  ", end='')

        # 记录损失
        train_losses.append(loss_dict['loss'])
        test_losses.append(error_dict['loss'])

        scheduler.step()

    if not os.path.exists('./trained_model'):
        os.makedirs('./trained_model')
    torch.save(model, f'trained_model/route_{model_config.obs_len}_{model_config.pred_len}.pth')


def route_extract(dataloader, model_config, param_config, seed, device, n_clusters):
    obs_len = model_config.obs_len
    pred_len = model_config.pred_len
    traj_len = obs_len + pred_len
    if not os.path.exists(f'./trained_model/route_{obs_len}_{pred_len}.pth'):
        train_ae(model_config, param_config, device)
    model = torch.load(f'./trained_model/route_{obs_len}_{pred_len}.pth', weights_only=False).to(device)
    traj_train, init_pos = dataloader.traj_train, dataloader.train_init_pos
    traj_train_ = []
    for traj in traj_train:
        for tra in traj:
            traj_train_.append(tra[:traj_len])
    init_train = []
    for init in init_pos:
        for ini in init:
            init_train.append(ini)

    traj_train = np.array(traj_train_)
    traj_train = torch.DoubleTensor(traj_train).to(device)
    init_train = np.array(init_train)
    init_train = torch.DoubleTensor(init_train).to(device)
    with torch.no_grad():
        traj_train_enc = model.encode(traj_train, init_train).cpu().numpy().squeeze()
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=10000, random_state=seed)
    kmeans.fit(traj_train_enc)  # 训练聚类模型
    return kmeans, model


def vis(n_clusters, cluster_labels, traj_train, rec_route):
    plt.figure(figsize=(15, 15))

    for cluster_id in range(n_clusters):
        plt.subplot(15, 10, cluster_id + 1)  # 调整子图的行数和列数以适应聚类数量
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_data = traj_train[cluster_indices].cpu().numpy()

        for traj in cluster_data:
            plt.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3)  # 绘制簇中所有轨迹

        rec_traj = rec_route[cluster_id]
        plt.plot(rec_traj[:, 0], rec_traj[:, 1], 'b-', linewidth=2)  # 绘制重构的轨迹

        plt.title(f'Cluster {cluster_id + 1}')
        plt.axis('equal')

    plt.tight_layout()
    plt.show()
