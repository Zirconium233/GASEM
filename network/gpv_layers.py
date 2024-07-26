# import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.adam as adam
import math
import numpy as np
import torch.optim as optim
# from datasets.datasets_pair import *
from datasets.GAPartNet.misc.info import OBJECT_NAME2ID

ID2OBJECT_NAME = {v: k for k, v in OBJECT_NAME2ID.items()}

# util function
def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int): # type: ignore
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


def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"): # type: ignore
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2))  # (bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim=2)  # (bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim=2)  # (bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k=1, dim=-1, largest=False)[1]
    return nearest_index


def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"): # type: ignore
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


def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"): # type: ignore
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    # ss = time.time()
    neighbors = indexing_neighbor(vertices, neighbor_index)  # (bs, v, n, 3)

    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    return neighbor_direction_norm.float()

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
                neighbor_index: "(bs, vertice_num, neighbor_num)", # type: ignore
                vertices: "(bs, vertice_num, 3)"): # type: ignore
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
                neighbor_index: "(bs, vertice_num, neighbor_index)", # type: ignore
                vertices: "(bs, vertice_num, 3)", # type: ignore
                feature_map: "(bs, vertice_num, in_channel)"): # type: ignore
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
                vertices: "(bs, vertice_num, 3)", # type: ignore
                feature_map: "(bs, vertice_num, channel_num)"): # type: ignore
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
                vertices: "tensor (bs, vetice_num, 3)", # type: ignore
                cat_id: "tensor (bs, 1)", # type: ignore
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

class PoseNet9D(nn.Module):
    def __init__(self):
        super(PoseNet9D, self).__init__()
        self.rot_green = Rot_green()
        self.rot_red = Rot_red()
        self.face_recon = FaceRecon()
        self.ts = Pose_Ts()

    def forward(self, points, obj_id):
        bs, p_num = points.shape[0], points.shape[1]
        recon, face, feat = self.face_recon(points - points.mean(dim=1, keepdim=True), obj_id)
        recon = recon + points.mean(dim=1, keepdim=True)
        # handle face
        face_normal = face[:, :, :18].view(bs, p_num, 6, 3)  # normal
        face_normal = face_normal / torch.norm(face_normal, dim=-1, keepdim=True)  # bs x nunm x 6 x 3
        face_dis = face[:, :, 18:24]  # bs x num x  6
        face_f = F.sigmoid(face[:, :, 24:])  # bs x num x 6
        #  rotation
        green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])

        # translation and size
        feat_for_ts = torch.cat([feat, points-points.mean(dim=1, keepdim=True)], dim=2)
        T, s = self.ts(feat_for_ts.permute(0, 2, 1))
        Pred_T = T + points.mean(dim=1)  # bs x 3
        Pred_s = s  # this s is not the object size, it is the residual

        return recon, face_normal, face_dis, face_f, p_green_R, p_red_R, f_green_R, f_red_R, Pred_T, Pred_s