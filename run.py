import glob
import os
import pickle
import shutil
import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
from dataset import DataLoader
from config import Config
from model.model import Model
from utils import custom_metrics, custom_criterion, set_input_targets, get_ade_fde_true, get_labels
from train_route_extract import route_extract


def train(model, optimizer, data_loader, args, criterion, device):
    model.train()  # 设置模型为训练模式
    loss_dict = {}
    traj_train, mask_train, train_labels, route, train_init_pos = data_loader.traj_train, data_loader.mask_train, data_loader.train_labels, data_loader.route, data_loader.train_init_pos
    for batch_idx, (traj, mask, label, init_pos) in enumerate(zip(traj_train, mask_train, train_labels, train_init_pos)):
        # 准备数据
        inputs, targets = set_input_targets(traj, mask, label, route, init_pos, args, device)

        # 清零优化器的梯度
        optimizer.zero_grad()

        outputs = model(inputs)

        res_dict = criterion(outputs, targets)

        for key, value in res_dict.items():
            if key not in loss_dict:
                loss_dict[key] = 0
            loss_dict[key] += value.item()

        loss = res_dict['loss']
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()

    for key, value in loss_dict.items():
        loss_dict[key] = value / len(traj_train)
    return loss_dict


def test(model, dataloader, epoch, criterion, device, model_args, param_args, outputdir):
    traj_test, mask_test, test_labels, route, test_init_pos = dataloader.traj_test, dataloader.mask_test, dataloader.test_labels, dataloader.route, dataloader.test_init_pos
    mmsi_test = dataloader.mmsi_test
    model.eval()
    metrics_dict = {}

    with torch.no_grad():
        for i, (traj, mask, label, init_pos, mmsi) in enumerate(zip(traj_test, mask_test, test_labels, test_init_pos, mmsi_test)):
            inputs, targets = set_input_targets(traj, mask, label, route, init_pos, model_args, device, mmsi)

            outputs = model.predict(inputs, num_k=param_args.num_k, route_num=param_args.route_num)

            args = param_args
            args.epoch = epoch
            args.batch_num = i
            if criterion.__name__ == 'custom_metrics':
                res_dict = criterion(outputs, targets, args)
            elif criterion.__name__ == 'get_ade_fde_true':
                res_dict = criterion(outputs, targets, args, outputdir)
            else:
                raise ValueError(f"Unknown criterion: {criterion.__name__}")
            for key, value in res_dict.items():
                if key not in metrics_dict:
                    metrics_dict[key] = 0
                metrics_dict[key] += value

    for key, value in metrics_dict.items():
        metrics_dict[key] = value / len(traj_test)
    return metrics_dict


def main(output_dir='./output'):
    '''
        1. 读取model参数和其他参数
        2. 设置数据集
        3. 设置train和test方法
    '''
    # 1. 读取model参数和其他参数
    model_config_src = "./configs/model.json"
    params_config_src = "./configs/params.json"
    model_config = Config(model_config_src)
    param_config = Config(params_config_src)

    args = parse_args()
    model_config.obs_len = args.obs_len
    model_config.pred_len = args.pred_len
    param_config.num_epochs = args.num_epochs

    if args.scene == "hainan":
        param_config.lon_min = 109
        param_config.lon_max = 109.2
        param_config.lat_min = 18.3
        param_config.lat_max = 18.4
        param_config.batches_path = './data/hainan_batches.pkl'
    elif args.scene == "zhoushan":
        param_config.lon_min = 122.2
        param_config.lon_max = 122.8
        param_config.lat_min = 30.2
        param_config.lat_max = 30.7
        param_config.batches_path = './data/s_batches.pkl'
    elif args.scene == "masked":
        param_config.lon_min = 92.215
        param_config.lon_max = 92.27498666666666
        param_config.lat_min = 0.20000333333333
        param_config.lat_max = 0.24997
        param_config.batches_path = './data/mask_batches.pkl'

    obs_len = model_config.obs_len
    pred_len = model_config.pred_len
    output_dir = output_dir + f'/{obs_len}_{pred_len}/'
    if os.path.exists(output_dir + '/vis_plots'):
        shutil.rmtree(output_dir + '/vis_plots')

    seed = param_config.random_state
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 2. 加载数据集
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    dataloader = DataLoader(param_config.batches_path, param_config.data_scale, param_config.lon_min, param_config.lon_max, param_config.lat_min, param_config.lat_max, param_config.test_size, param_config.random_state, model_config.obs_len, model_config.pred_len)

    # for i, _ in enumerate(dataloader.traj_train):
    #     dataloader.traj_train[i] = dataloader.traj_train[i][:, :traj_len]
    # for i, _ in enumerate(dataloader.traj_test):
    #     dataloader.traj_test[i] = dataloader.traj_test[i][:, :traj_len]

    kmeans, ae = route_extract(dataloader, model_config, param_config, param_config.random_state, device, param_config.n_clusters)

    train_labels, test_labels = get_labels(dataloader, kmeans, ae, device)

    dataloader.train_labels = train_labels
    dataloader.test_labels = test_labels

    route = ae.decode(torch.DoubleTensor(kmeans.cluster_centers_).to(device)).cpu().detach().numpy()

    dataloader.route = route

    epochs = param_config.num_epochs

    # 3. 创建model
    curr_epoch = 1
    check_save_path = output_dir + "/check_points/"
    if not os.path.exists(check_save_path):
        os.makedirs(check_save_path)
    model_save_path = output_dir + "/saved_models/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    files = []
    if os.path.exists(check_save_path):
        files = glob.glob(check_save_path + "*.pth")
    if len(files) == 0:
        model = Model(model_config).to(device)
    else:
        check_epoch = []
        for file in files:
            check_epoch.append(int(file.split('/')[-1].split('.')[0].split('_')[-1]))
        max_check = max(check_epoch)
        curr_epoch = max_check
        model = torch.load(check_save_path + f"model_{max_check}.pth", weights_only=False).to(device)
    del files
    model.double()
    optimizer = Adam(lr=param_config.learning_rate, params=model.parameters())
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=param_config.lr_milestones, gamma=param_config.gama)
    scheduler.step(curr_epoch)
    best_epoch = 0
    min_ade = 9999
    min_fde = 9999
    min_true_ade = 9999
    min_true_fde = 9999
    # 4. 训练的 epoch
    for epoch in range(curr_epoch - 1, epochs):
        loss_dict = train(model, optimizer, dataloader, model_config, custom_criterion, device)
        loss_str = ""
        for key in loss_dict:
            loss_str += f"{key} = {loss_dict[key]}   "

        metrics_dict = test(model, dataloader, epoch, custom_metrics, device, model_config, param_config, output_dir)
        ade = metrics_dict['min_ade']
        fde = metrics_dict['min_fde']

        if ade + fde < min_ade + min_fde:
            min_ade = ade
            min_fde = fde
            best_epoch = epoch
            true_metrics_dict = test(model, dataloader, epoch, get_ade_fde_true, device, model_config, param_config, output_dir + '/vis_plots')
            true_ade = true_metrics_dict['min_ade']
            true_fde = true_metrics_dict['min_fde']
            min_true_ade = true_ade
            min_true_fde = true_fde
            torch.save(model, model_save_path + 'model.pth')
        if (epoch + 1) % param_config.check == 0:
            torch.save(model, check_save_path + f'model_{epoch + 1}.pth')
        scheduler.step()

        print(f"\r[{args.scene} {model_config.obs_len} {model_config.pred_len}][Epoch {epoch + 1}] [Best Epoch {best_epoch + 1}] [Best ADE {min_true_ade}] [Best FDE {min_true_fde}] {loss_str} ", end='')


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--obs_len', type=int, default=6, help='observation length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction length')
    parser.add_argument('--num_epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--scene', type=str, default="mask", help='dataset')
    parser.add_argument('--cuda', type=int, default=0, help='cuda device')
    return parser.parse_args()


if __name__ == '__main__':
    main()