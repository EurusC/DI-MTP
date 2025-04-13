import torch
import torch.nn as nn
import torch.nn.functional as F
from model.component import MLP, TrajEncoder, PositionalEncoding, SelfAttention
import sys
from torch.distributions import Normal, Categorical

sys.path.append('../')


'''
    模型思想：
        1. 航道提取模块：AE自编码，自学习轨迹特征，用kmeans提取具体航道（提取聚类中心）
        2. 航道提取后，构建观察轨迹与航道数据之间的映射关系，对kmeans的label结果，在数据中给出航道信息列
        3. 构建模型，实现观察轨迹与航道信息之间的分类任务，与根据航道信息实现轨迹修改的轨迹修改任务（轨迹预测）
        4. 具体实现细节可以是：观察轨迹和航道信息之间的分类任务；tf下根据给定的航道标签和对应的航道信息，提取对应的航道编码enc，作为输入条件之一
        5. continue：多模态性可以认为是航道修改上的多模态性，这个多模态性的构建可以通过CVAE构建，根据obs_enc,航道编码enc和轨迹之间的相互注意力，生成第一步的修改轨迹编码
        6. continue：对最后的船舶轨迹，non-local att，保证全局轨迹的正确性，最终解码为预测轨迹
        亮点：针对船舶轨迹特性的模型设计（航道信息、船舶彼此间的社交关系、船舶轨迹的重复性）；预训练机制，针对不同地区的自适应能力；
'''


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()

        self.traj_len = args.obs_len + args.pred_len
        self.enc_dim = args.enc_dim
        self.feat_dims = args.feat_dims
        self.hidden = args.hidden
        self.dropout = args.dropout_rate

        self.traj_encoder = nn.Sequential(
            MLP(self.traj_len * self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.enc_dim),
        )

        self.encoder = nn.Sequential(
            MLP(self.enc_dim + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.enc_dim),
        )

        self.decoder = nn.Sequential(
            MLP(self.enc_dim, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.traj_len * self.feat_dims),
        )

    def forward(self, traj, init_pos):
        traj_enc = self.encode(traj, init_pos)
        rec = self.decode(traj_enc)
        res = {'rec': rec}
        return res

    def encode(self, traj, init_pos):
        traj = traj.reshape(traj.shape[0], 1, self.traj_len * self.feat_dims)
        traj_enc = self.traj_encoder(traj)
        feats = torch.cat([traj_enc, init_pos], dim=-1)
        traj_enc = self.encoder(feats)
        return traj_enc

    def decode(self, enc):
        traj = self.decoder(enc)
        traj = traj.reshape(traj.shape[0], self.traj_len, self.feat_dims)
        return traj


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.enc_dim = args.enc_dim
        self.att_hidden = args.att_hidden
        self.att_layer = args.att_layer
        self.dropout = args.dropout_rate
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.feat_dims = args.feat_dims
        self.hidden = args.hidden
        self.sigma = args.sigma

        # obs_traj 2 obs_enc
        self.obs_encoder = nn.Sequential(
            MLP(self.obs_len * self.feat_dims + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.enc_dim),
        )

        self.route_enc = nn.Sequential(
            MLP((self.obs_len + self.pred_len) * self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.enc_dim),
        )

        # modify_route_end 2 route_end_enc
        self.route_end_enc = nn.Sequential(
            MLP(self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.enc_dim),
        )

        # self attention for interaction
        self.int_att = nn.ModuleList(
            [SelfAttention(in_size=self.enc_dim, hidden_size=self.hidden[2], out_size=self.enc_dim) for _ in range(self.att_layer)]
        )

        self.scoring_att = SelfAttention(in_size=self.enc_dim, hidden_size=self.hidden[2], out_size=self.enc_dim)

        # [obs_enc, route_enc] 2 latent
        self.latent = nn.Sequential(
            MLP(3 * self.enc_dim + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], 2 * self.enc_dim),
        )

        # z 2 pred_modify_route_end
        self.latent_decoder = nn.Sequential(
            MLP(3 * self.enc_dim + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.feat_dims),
        )

        # feats 2 pred_traj
        self.decoder = nn.Sequential(
            MLP(self.enc_dim * 3 + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            MLP(self.hidden[2], self.feat_dims * self.pred_len),
        )

        self.non_local_theta = nn.Sequential(
            MLP(self.enc_dim * 3 + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            nn.Linear(self.hidden[2], 1024)
        )

        self.non_local_phi = nn.Sequential(
            MLP(self.enc_dim * 3 + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            nn.Linear(self.hidden[2], 1024)
        )

        self.non_local_g = nn.Sequential(
            MLP(self.enc_dim * 3 + self.feat_dims, self.hidden[0]),
            MLP(self.hidden[0], self.hidden[1]),
            MLP(self.hidden[1], self.hidden[2]),
            nn.Linear(self.hidden[2], self.enc_dim * 3 + self.feat_dims)
        )

    def global_rational(self, feat, mask):
        theta = self.non_local_theta(feat).permute(1, 0, 2)
        phi = self.non_local_phi(feat).permute(1, 2, 0)
        f = torch.matmul(theta, phi)
        f_weights = F.softmax(f, dim=-1)
        f_weights = f_weights * mask
        f_weights = F.normalize(f_weights, p=1, dim=1)
        pooled_f = torch.matmul(f_weights, self.non_local_g(feat).permute(1, 0, 2))
        pooled_f = pooled_f.permute(1, 0, 2)
        return pooled_f + feat

    def forward(self, inputs):
        res_dict = {}
        obs_traj = inputs.obs_traj
        mask = inputs.mask
        device = inputs.device
        gt_traj = inputs.gt_traj
        route = inputs.route
        label = inputs.label
        init_pos = inputs.init_pos

        # obs_enc 编码
        obs_traj = obs_traj.reshape(obs_traj.shape[0], 1, -1)
        obs_enc = self.obs_encoder(torch.cat([obs_traj, init_pos], dim=-1))

        # modify_route_end 编码
        modify_route_end = gt_traj[:, -2:-1, :]
        gt_end_enc = self.route_end_enc(modify_route_end)

        # route_enc 编码
        route = route.reshape(route.shape[0], 1, -1)
        route_enc = self.route_enc(route)
        route_enc = route_enc.reshape(1, route_enc.shape[0], -1)
        route_enc_ = route_enc.repeat(obs_enc.shape[0], 1, 1)
        route_enc = route_enc.squeeze()

        # 创建obs_enc和航道label之间的分类问题
        # obs_enc彼此之间的相互关系
        obs_enc = obs_enc.permute(1, 0, 2)  # 1 N enc_dim
        for i in range(len(self.int_att)):
            int_mat = self.int_att[i](obs_enc, obs_enc, mask)
            obs_enc = obs_enc + torch.matmul(int_mat, obs_enc)
        obs_enc = obs_enc.permute(1, 0, 2)  # N 1 enc_dim

        # 构建obs_enc与route_enc之间的关联，作为分类任务
        path_score = self.scoring_att(obs_enc, route_enc_).squeeze()
        res_dict['path_score'] = path_score

        # 取得对应的route_enc
        closet_route_enc = route_enc[label].unsqueeze(1)

        # 对航道修改的多模态性进行建模，构建cvae
        # 对航道修改的方式，我们要有自己给出的定义，现在的模型对于航道修改这块的效果不好，最好在航道修改这块，加上一定的条件限制，不然效果应该只能一般
        latent_feats = torch.cat([obs_enc, closet_route_enc, init_pos, gt_end_enc], dim=-1)
        latent = self.latent(latent_feats)
        mu, log = latent[:, :, :self.enc_dim], latent[:, :, self.enc_dim:]
        res_dict['mu'] = mu
        res_dict['log'] = log
        var = log.mul(0.5).exp_()
        eps = torch.DoubleTensor(var.size()).normal_()
        eps = eps.to(device)
        z = eps.mul(var).add_(mu)
        z = z.double().to(device)

        # 解码得到粗糙编码
        latent_dec_feats = torch.cat([obs_enc, closet_route_enc, init_pos, z], dim=-1)
        pred_modify_route_end = self.latent_decoder(latent_dec_feats)
        res_dict['pred_route_end'] = pred_modify_route_end

        pred_end_enc = self.route_end_enc(pred_modify_route_end)

        # non-local att
        pred_feats = torch.cat([obs_enc, closet_route_enc, pred_end_enc, init_pos], dim=-1)

        for _ in range(self.att_layer):
            pred_feats = self.global_rational(pred_feats, mask)

        pred_traj = self.decoder(pred_feats)
        res_dict['pred_traj'] = pred_traj
        return res_dict

    def predict(self, inputs, num_k=50, route_num=10):
        res_dict = {}
        obs_traj = inputs.obs_traj
        mask = inputs.mask
        device = inputs.device
        batch_size = obs_traj.shape[0]
        route = inputs.route
        init_pos = inputs.init_pos
        label = inputs.label

        # obs_enc 编码
        obs_traj = obs_traj.reshape(batch_size, 1, -1)
        obs_enc = self.obs_encoder(torch.cat([obs_traj, init_pos], dim=-1))

        # route_enc 编码
        route = route.reshape(route.shape[0], 1, -1)
        route_enc = self.route_enc(route)
        route_enc = route_enc.reshape(1, route_enc.shape[0], -1)
        route_enc_ = route_enc.repeat(obs_enc.shape[0], 1, 1)

        obs_enc = obs_enc.permute(1, 0, 2)
        for i in range(len(self.int_att)):
            int_mat = self.int_att[i](obs_enc, obs_enc, mask)
            obs_enc = obs_enc + torch.matmul(int_mat, obs_enc)
        obs_enc = obs_enc.permute(1, 0, 2)

        # 获取对应的route_enc
        path_score = self.scoring_att(obs_enc, route_enc_).squeeze()
        top_10_indices = torch.topk(path_score, route_num, dim=-1).indices

        temperature = 2.0  # 调整温度参数
        prob = torch.softmax(path_score / temperature, dim=-1)

        top_10_prob = prob.gather(dim=-1, index=top_10_indices)
        res_dict['prob'] = top_10_prob

        ped_indices = torch.arange(0, obs_enc.shape[0]).unsqueeze(1).to(device)
        selected_paths_enc = route_enc_[ped_indices, top_10_indices]

        assert num_k % route_num == 0
        num_k_ = int(num_k / route_num)
        # 构建cvae建模轨迹的修改
        z = torch.DoubleTensor(batch_size, route_num, num_k_, self.enc_dim).to(device)
        z.normal_(0, self.sigma)
        obs_enc = obs_enc.unsqueeze(1)
        obs_enc = obs_enc.repeat(1, route_num, num_k_, 1)
        init_pos = init_pos.reshape(init_pos.shape[0], 1, 1, -1)
        init_pos = init_pos.repeat(1, route_num, num_k_, 1)
        # if selected_paths_enc.shape[1] == 1:
        #     label = label.unsqueeze(1)
        #     selected_paths_enc = route_enc_[ped_indices, label]
        #     selected_paths_enc = selected_paths_enc.unsqueeze(2).repeat(1, route_num, num_k_, 1)
        # else:
        #     selected_paths_enc = selected_paths_enc.unsqueeze(2).repeat(1, 1, num_k_, 1)
        selected_paths_enc = selected_paths_enc.unsqueeze(2).repeat(1, 1, num_k_, 1)
        # 解码得到 coarse_enc
        latent_dec_feats = torch.cat([obs_enc, selected_paths_enc, init_pos, z], dim=-1)
        pred_route_end = self.latent_decoder(latent_dec_feats)
        pred_end_enc = self.route_end_enc(pred_route_end)

        # non-local att
        pred_feats = torch.cat([obs_enc, selected_paths_enc, pred_end_enc, init_pos], dim=-1)
        mask = mask.repeat(route_num, route_num)
        pred_feats_shape = pred_feats.shape
        pred_feats = pred_feats.reshape(-1, num_k_, pred_feats.shape[-1])
        for _ in range(self.att_layer):
            pred_feats = self.global_rational(pred_feats, mask)
        pred_feats = pred_feats.reshape(pred_feats_shape)
        pred_traj = self.decoder(pred_feats)
        pred_traj = pred_traj.reshape(pred_traj.shape[0], num_k, self.pred_len, self.feat_dims)
        res_dict['pred_traj'] = pred_traj
        return res_dict


    def get_z(self, inputs):
        res_dict = {}
        obs_traj = inputs.obs_traj
        mask = inputs.mask
        device = inputs.device
        gt_traj = inputs.gt_traj
        route = inputs.route
        label = inputs.label
        init_pos = inputs.init_pos

        # obs_enc 编码
        obs_traj = obs_traj.reshape(obs_traj.shape[0], 1, -1)
        obs_enc = self.obs_encoder(torch.cat([obs_traj, init_pos], dim=-1))

        # modify_route_end 编码
        modify_route_end = gt_traj[:, -2:-1, :]
        gt_end_enc = self.route_end_enc(modify_route_end)

        # route_enc 编码
        route = route.reshape(route.shape[0], 1, -1)
        route_enc = self.route_enc(route)
        route_enc = route_enc.reshape(1, route_enc.shape[0], -1)
        route_enc_ = route_enc.repeat(obs_enc.shape[0], 1, 1)
        route_enc = route_enc.squeeze()

        # 创建obs_enc和航道label之间的分类问题
        # obs_enc彼此之间的相互关系
        obs_enc = obs_enc.permute(1, 0, 2)  # 1 N enc_dim
        for i in range(len(self.int_att)):
            int_mat = self.int_att[i](obs_enc, obs_enc, mask)
            obs_enc = obs_enc + torch.matmul(int_mat, obs_enc)
        obs_enc = obs_enc.permute(1, 0, 2)  # N 1 enc_dim

        # 构建obs_enc与route_enc之间的关联，作为分类任务
        path_score = self.scoring_att(obs_enc, route_enc_).squeeze()
        res_dict['path_score'] = path_score

        # 取得对应的route_enc
        closet_route_enc = route_enc[label].unsqueeze(1)

        # 对航道修改的多模态性进行建模，构建cvae
        # 对航道修改的方式，我们要有自己给出的定义，现在的模型对于航道修改这块的效果不好，最好在航道修改这块，加上一定的条件限制，不然效果应该只能一般
        latent_feats = torch.cat([obs_enc, closet_route_enc, init_pos, gt_end_enc], dim=-1)
        latent = self.latent(latent_feats)
        mu, log = latent[:, :, :self.enc_dim], latent[:, :, self.enc_dim:]
        res_dict['mu'] = mu
        res_dict['log'] = log
        var = log.mul(0.5).exp_()
        eps = torch.DoubleTensor(var.size()).normal_()
        eps = eps.to(device)
        z = eps.mul(var).add_(mu)
        z = z.double().to(device)
        return z