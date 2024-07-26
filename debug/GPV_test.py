# import 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam as adam
import math
import numpy as np
from torchinfo import summary
import torch.optim as optim
import functools
from torch.autograd import Variable
import open3d.ml.torch as ml3d
from datasets.datasets_pair import *
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
from datetime import datetime
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# util function
def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2))  # (bs, v, v)
    quadratic = torch.sum(vertices ** 2, dim=2)  # (bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """

    bs, v, n = index.size()

    # ss = time.time()
    if bs == 1:
        # id_0 = torch.arange(bs).view(-1, 1,1)
        tensor_indexed = tensor[torch.Tensor([[0]]).long(), index[0]].unsqueeze(dim=0)
    else:
        id_0 = torch.arange(bs).view(-1, 1, 1).long()
        tensor_indexed = tensor[id_0, index]
    # ee = time.time()
    # print('tensor_indexed time: ', str(ee - ss))
    return tensor_indexed


def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    # ss = time.time()
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)

    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    return neighbor_direction_norm.float()


def get_gt_v(Rs, axis=2):
    bs = Rs.shape[0]  # bs x 3 x 3
    # TODO use 3 axis, the order remains: do we need to change order?
    if axis == 3:
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    else:
        assert axis == 2
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    gt_green = gt_vec[:, 3:6]
    gt_red = gt_vec[:, (6, 7, 8)]
    return gt_green, gt_red

# gcn3d layers
class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""

    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace=True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        # ss = time.time()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        # R = get_rotation(0,0,0)
        # R = torch.from_numpy(R).cuda()
        # R = R.unsqueeze(0).repeat(bs,1,1).float() ## bs 3,3
        # vertices2 = torch.bmm(R,vertices.transpose(1,2)).transpose(2,1)
        # neighbor_direction_norm2 = get_neighbor_direction_norm(vertices2, neighbor_index)

        support_direction_norm = F.normalize(self.directions, dim=0)  # (3, s * k)

        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim=2)[0]  # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim=2)  # (bs, vertice_num, kernel_num)
        return feature


class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace=True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim=0)
        theta = neighbor_direction_norm @ support_direction_norm  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias  # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel]  # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:]  # (bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support,
                                            neighbor_index)  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support  # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs, vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim=2)[0]  # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim=2)  # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support  # (bs, vertice_num, out_channel)
        return feature_fuse


class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map,
                                             neighbor_index)  # (bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]  # (bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]  # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :]  # (bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool
    
# posenet layers
class Rot_green(nn.Module):
    def __init__(self, feat_c_R, R_c):
        super(Rot_green, self).__init__()
        self.f = feat_c_R
        self.k = R_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


class Rot_red(nn.Module):
    def __init__(self, feat_c_R, R_c):
        super(Rot_red, self).__init__()
        self.f = feat_c_R
        self.k = R_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


class Pose_Ts(nn.Module):
    def __init__(self, feat_c_ts, Ts_c):
        super(Pose_Ts, self).__init__()
        self.f = feat_c_ts
        self.k = Ts_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs
    

class FaceRecon(nn.Module):
    def __init__(self, gcn_n_num, gcn_sup_num, face_recon_c, obj_c, feat_face):
        super(FaceRecon, self).__init__()
        self.neighbor_num = gcn_n_num
        self.support_num = gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3 = Conv_layer(256, 256, support_num=self.support_num)
        self.pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = Conv_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.recon_num = 3
        self.face_recon_num = face_recon_c
        self.obj_c = obj_c
        
        dim_fuse = sum([128, 128, 256, 256, 512, obj_c])
        # 16: total 6 categories, 256 is global feature
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.recon_head = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.recon_num, 1),
        )

        self.face_head = nn.Sequential(
            nn.Conv1d(feat_face + 3, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.face_recon_num, 1),  # Relu or not?
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                ):
        """
        Return: (bs, vertice_num, class_num)
        """
        #  concate feature
        bs, vertice_num, _ = vertices.size()
        # cat_id to one-hot
        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, self.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)
        # bs x verticenum x 6

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        # ss = time.time()
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        # neighbor_index = get_neighbor_index(v_pool_1, self.neighbor_num)
        neighbor_index = get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        # neighbor_index = get_neighbor_index(v_pool_2, self.neighbor_num)
        neighbor_index = get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        f_global = fm_4.max(1)[0]  # (bs, f)

        nearest_pool_1 = get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = get_nearest_index(vertices, v_pool_2)
        fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)  # (bs, vertice_num, cat_one_hot)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''
        # feat_face_re = self.global_perception_head(feat)  # bs x C x 1
        feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1)
        # feat is the extracted per pixel level feature

        conv1d_input = feat.permute(0, 2, 1)  # (bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)

        recon = self.recon_head(conv1d_out)
        # average pooling for face prediction
        feat_face_in = torch.cat([feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1)
        face = self.face_head(feat_face_in)
        return recon.permute(0, 2, 1), face.permute(0, 2, 1), feat
    
# postnet9d
class FaceRecon_feat(nn.Module):
    def __init__(self, gcn_n_num, gcn_sup_num):
        super(FaceRecon_feat, self).__init__()
        self.neighbor_num = gcn_n_num
        self.support_num = gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3 = Conv_layer(256, 256, support_num=self.support_num)
        self.pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = Conv_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                # cat_id: "tensor (bs, 1)",
                ):
        """
        Return: (bs, vertice_num, class_num)
        """

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        neighbor_index = get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        nearest_pool_1 = get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = get_nearest_index(vertices, v_pool_2)
        fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''
        return feat

class PoseNet9D_Only_R(nn.Module):
    def __init__(self, feat_c_R=1280, R_c=4, gcn_n_num=10, gcn_sup_num=7, face_recon_c=6 * 5, obj_c=6, feat_face=768, feat_c_ts=1289, Ts_c=6):
        super(PoseNet9D_Only_R, self).__init__()
        self.rot_green = Rot_green(feat_c_R, R_c)
        self.rot_red = Rot_red(feat_c_R, R_c)
        self.face_recon = FaceRecon_feat(gcn_n_num, gcn_sup_num)
        # self.ts = Pose_Ts(feat_c_ts, Ts_c)

    def forward(self, points):
        bs, p_num = points.shape[0], points.shape[1]
        feat = self.face_recon(points - points.mean(dim=1, keepdim=True))
        #  rotation
        green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])
        # translation and size no need
        return p_green_R, p_red_R, f_green_R, f_red_R
    
# loss
class fs_net_loss_R(nn.Module):
    def __init__(self, loss_type="smoothl1"):
        super(fs_net_loss_R, self).__init__()
        if loss_type == 'l1':
            self.loss_func_t = nn.L1Loss()
            self.loss_func_s = nn.L1Loss()
            self.loss_func_Rot1 = nn.L1Loss()
            self.loss_func_Rot2 = nn.L1Loss()
            self.loss_func_r_con = nn.L1Loss()
            self.loss_func_Recon = nn.L1Loss()
        elif loss_type == 'smoothl1':   # same as MSE
            self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot1 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot2 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_r_con = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Recon = nn.SmoothL1Loss(beta=0.3)
        else:
            raise NotImplementedError

    def forward(self, pred_list, gt_list):
        loss_list = {}

        self.rot_1_w = 1

        loss_list["Rot1"] = self.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_list["Rot1"])

        loss_list["Rot2"] = self.rot_1_w * self.cal_loss_Rot1(pred_list["Rot2"], gt_list["Rot2"])

        # loss_list["Recon"] = self.recon_w * self.cal_loss_Recon(pred_list["Recon"], gt_list["Recon"])

        # loss_list["Tran"] = self.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])
    
        # loss_list["Size"] = self.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])

        return loss_list

    def cal_loss_Rot1(self, pred_v, gt_v):
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
        for i in range(bs):
            pred_v_now = pred_v[i, ...]
            gt_v_now = gt_v[i, ...]
            res[i] = self.loss_func_Rot1(pred_v_now, gt_v_now)
        res = torch.mean(res)
        return res

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += self.loss_func_Rot2(pred_v_now, gt_v_now)
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_Recon(self, pred_recon, gt_recon):
        return self.loss_func_Recon(pred_recon, gt_recon)

    def cal_loss_Tran(self, pred_trans, gt_trans):
        return self.loss_func_t(pred_trans, gt_trans)

    def cal_loss_Size(self, pred_size, gt_size):
        return self.loss_func_s(pred_size, gt_size)

root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
test_intra_dir = "/16T/zhangran/GAPartNet_re_rendered/test_intra"
test_inter_dir = "/16T/zhangran/GAPartNet_re_rendered/test_inter"
dataset_train = GAPartNetPair(
                    Path(root_dir)  / "pth",
                    Path(root_dir)  / "meta",
                    shuffle=True,
                    max_points=2000,
                    augmentation=True,
                    voxelization=False, 
                    group_size=2,
                    voxel_size=[0.01,0.01,0.01],
                    few_shot=False,
                    few_shot_num=None,
                    # few_shot = True,
                    # few_shot_num = 20,
                    pos_jitter=0.1,
                    with_pose=True,
                    color_jitter=0.3,
                    flip_prob=0.3,
                    rotate_prob=0.3,
                )
dataloader_train = DataLoader(
                    dataset_train,
                    batch_size=16,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=data_utils.trivial_batch_collator,
                    pin_memory=True,
                    drop_last=False,
                )
dataset_test_intra = GAPartNetPair(
                    Path(test_intra_dir)  / "pth",
                    Path(test_intra_dir)  / "meta",
                    shuffle=False,
                    max_points=2000,
                    augmentation=True,
                    voxelization=False, 
                    group_size=2,
                    voxel_size=[0.01,0.01,0.01],
                    few_shot=False,
                    few_shot_num=None,
                    # few_shot = True,
                    # few_shot_num = 20,
                    pos_jitter=0.1,
                    with_pose=True,
                    color_jitter=0.3,
                    flip_prob=0.3,
                    rotate_prob=0.3,
                )
dataloader_test_intra = DataLoader(
                    dataset_test_intra,
                    batch_size=16,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=data_utils.trivial_batch_collator,
                    pin_memory=True,
                    drop_last=False,
                )
dataset_test_inter = GAPartNetPair(
                    Path(test_inter_dir)  / "pth",
                    Path(test_inter_dir)  / "meta",
                    shuffle=False,
                    max_points=2000,
                    augmentation=True,
                    voxelization=False, 
                    group_size=2,
                    voxel_size=[0.01,0.01,0.01],
                    few_shot=False,
                    few_shot_num=None,
                    # few_shot = True,
                    # few_shot_num = 20,
                    pos_jitter=0.1,
                    with_pose=True,
                    color_jitter=0.3,
                    flip_prob=0.3,
                    rotate_prob=0.3,
                )
dataloader_test_inter = DataLoader(
                    dataset_test_inter,
                    batch_size=16,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=data_utils.trivial_batch_collator,
                    pin_memory=True,
                    drop_last=False,
                )

class test_GPV(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PoseNet9D_Only_R()
    
    def forward(self, pc_list: List[PointCloudPair]):
        points1 = torch.cat([pc.pc1.points.unsqueeze(0) for pc in pc_list], dim=0).cuda()  # pc_list is batch size
        points2 = torch.cat([pc.pc2.points.unsqueeze(0) for pc in pc_list], dim=0).cuda()
        p_green_R1, p_red_R1, f_green_R1, f_red_R1 = self.backbone(points1[:, :, 0:3])
        p_green_R2, p_red_R2, f_green_R2, f_red_R2 = self.backbone(points2[:, :, 0:3])
        return (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2)

def vectors_to_rotation_matrix(green_vector, red_vector):
    # green_vector and red_vector are normalized
    green_vector = green_vector / torch.norm(green_vector, dim=1, keepdim=True)
    red_vector = red_vector / torch.norm(red_vector, dim=1, keepdim=True)
    blue_vector = torch.cross(green_vector, red_vector)
    
    rotation_matrix = torch.stack([red_vector, green_vector, blue_vector], dim=2)
    return rotation_matrix

def calculate_metrics(predictions, ground_truths):
    # Accuracy can be calculated as the mean of correct predictions
    accuracy = (predictions.argmax(dim=1) == ground_truths.argmax(dim=1)).float().mean().item()

    # Additional metrics using torchmetrics library
    precision = torchmetrics.functional.precision(predictions, ground_truths.argmax(dim=1), average='macro', num_classes=predictions.shape[1])
    recall = torchmetrics.functional.recall(predictions, ground_truths.argmax(dim=1), average='macro', num_classes=predictions.shape[1])
    f1 = torchmetrics.functional.f1_score(predictions, ground_truths.argmax(dim=1), average='macro', num_classes=predictions.shape[1])

    return accuracy, precision, recall, f1

def rotation_matrix_to_euler_angles(Rt):
    r = R.from_matrix(Rt.cpu().numpy())
    return r.as_euler('xyz', degrees=True)

def calculate_pose_metrics(pred_rot_matrices, gt_rot_matrices, pred_translations):
    batch_size = pred_rot_matrices.size(0)

    rot_errors = []
    for i in range(batch_size):
        pred_euler = rotation_matrix_to_euler_angles(pred_rot_matrices[i])
        gt_euler = rotation_matrix_to_euler_angles(gt_rot_matrices[i])
        rot_error = torch.tensor(pred_euler - gt_euler).abs().mean().item()
        rot_errors.append(rot_error)
    mean_rot_error = sum(rot_errors) / batch_size
    return mean_rot_error

# Helper function to extract ground truth rotation vectors from the batch of PointCloudPairs
def ground_truth_rotations(rot_list: List[torch.Tensor]) -> np.ndarray:
    rotations = []
    for rot in rot_list:
        # Assuming the rotations are stored as 3x3 matrices in pc_pair.rot_1 and pc_pair.rot_2
        rotation_matrix = np.array(rot.cpu())  # Example using rot_1, adjust as needed
        rotations.append(rotation_matrix)
    return torch.tensor(np.stack(rotations))

def train(model: test_GPV, dataloader_train, dataloader_test_inter, dataloader_test_intra, lr, num_epochs, log_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = fs_net_loss_R()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    log_dir = log_dir + "/" + str(datetime.today())
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    print("_________________________train_epoch___________________________")
    for epoch in range(num_epochs):
        total_loss = 0
        if epoch == 0:
            # first test epoch
            print("______________________first_test_epoch_________________________")
            torch.save(model.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_metrics(model, dataloader_test_inter, device, writer, epoch, 'test_inter')
            test_metrics(model, dataloader_test_intra, device, writer, epoch, 'test_intra')
        for batch_idx, batch in enumerate(dataloader_train):
            pc_pairs = [pair.to(device) for pair in batch]
            optimizer.zero_grad()

            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)
            
            # Assuming we have ground truth rotations
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
            pred_list1 = {
                "Rot1": p_green_R1,
                "Rot2": p_red_R1,
            }
            gt_list1 = {
                "Rot1": R_green_gt1.cuda(),
                "Rot2": R_red_gt1.cuda(),
            }
            
            pred_list2 = {
                "Rot1": p_green_R2,
                "Rot2": p_red_R2,
            }
            gt_list2 = {
                "Rot1": R_green_gt2.cuda(),
                "Rot2": R_red_gt2.cuda(),
            }

            loss_dict1 = criterion(pred_list1, gt_list1)
            loss_dict2 = criterion(pred_list2, gt_list2)
            loss = (loss_dict1['Rot1'] + loss_dict1['Rot2'] + loss_dict2['Rot1'] + loss_dict2['Rot2']) / 2.0
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # 每10个batch记录一次loss
            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)

        # 每5个epoch跑一次测试集
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_metrics(model, dataloader_test_inter, device, writer, epoch, 'test_inter')
            test_metrics(model, dataloader_test_intra, device, writer, epoch, 'test_intra')


def test_metrics(model, dataloader, device, writer, epoch, phase):
    print("______________________" + phase + "_______________________")
    model.eval()
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    all_pred_translations = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pc_pairs = [pair.to(device) for pair in batch]
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)
            
            # Assuming we have ground truth rotations
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
            # Convert predicted vectors and ground truth vectors back to rotation matrices
            pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1)
            pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2)
            gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1)
            gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2)
            
            # Store predictions and ground truths for metrics calculation
            all_pred_rot_matrices.append(pred_rot_matrices1.cpu())
            all_pred_rot_matrices.append(pred_rot_matrices2.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices1.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices2.cpu())
            all_pred_translations.append(f_green_R1.cpu())
            all_pred_translations.append(f_green_R2.cpu())
    
    all_pred_rot_matrices = torch.cat(all_pred_rot_matrices, dim=0)
    all_gt_rot_matrices = torch.cat(all_gt_rot_matrices, dim=0)
    all_pred_translations = torch.cat(all_pred_translations, dim=0)

    mean_rot_error = calculate_pose_metrics(
        all_pred_rot_matrices, all_gt_rot_matrices, all_pred_translations
    )
    writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
    print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    model.train()

if __name__ == "__main__":
    model = test_GPV()
    train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, 0.001, 40, "./log_dir/GPV_test")
    torch.save(model.state_dict(), "/home/zhangran/desktop/tmp/backup.pth")