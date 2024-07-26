import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d.ml.torch as ml3d
from datasets.datasets_pair import *
import spconv.pytorch as spconv
import torch.optim as optim
import functools
from torch.autograd import Variable
import argparse

root_dir = "./datasets/GAPartNet/dataset/data/"
def feature_transform_reguliarzer(trans):
    d = trans.size()[1] # k (bs, k, k)
    I = torch.eye(d)[None, :, :] # no batch size
    if trans.is_cuda:
        I = I.cuda() # to cuda
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2))) # 尽可能满足正交性质
    return loss

def pixel_accuracy(pred, label):
    correct = (pred == label).sum().item()
    total = label.size  # 修改这里，从label.numel()改为label.size
    return correct / total

def mean_pixel_accuracy(pred, label, num_classes):
    class_accuracies = []
    for c in range(num_classes):
        class_mask = (label == c)
        if class_mask.sum().item() == 0:
            continue
        class_accuracy = (pred[class_mask] == c).sum().item() / class_mask.sum().item()
        class_accuracies.append(class_accuracy)
    return np.mean(class_accuracies)

def intersection_over_union(pred, label, num_classes):
    ious = []
    for c in range(num_classes):
        pred_class = (pred == c)
        label_class = (label == c)
        intersection = (pred_class & label_class).sum().item()
        union = (pred_class | label_class).sum().item()
        if union == 0:
            ious.append(float('nan'))  # 如果没有出现这个类，则忽略
        else:
            ious.append(intersection / union)
    return np.array(ious)

def mean_intersection_over_union(pred, label, num_classes):
    ious = intersection_over_union(pred, label, num_classes)
    return np.nanmean(ious)  # 忽略NaN值

def frequency_weighted_intersection_over_union(pred, label, num_classes):
    ious = intersection_over_union(pred, label, num_classes)
    total = label.size  # 修改这里，从label.numel()改为label.size
    class_freq = np.array([(label == c).sum().item() / total for c in range(num_classes)])
    return (class_freq * ious).sum()

# 示例调用
# 假设 seg_1 是模型输出，labels_1 是标签
def evaluate_segmentation_metrics(seg_1, labels_1):
    # 将seg_1转化为预测标签
    pred = torch.argmax(seg_1, dim=1)

    # 将GPU tensor转化为CPU tensor，并转化为numpy数组
    pred = pred.cpu().numpy()
    labels_1 = labels_1.cpu().numpy()

    num_classes = 2  # 假设只有两个类：背景和前景

    pa = pixel_accuracy(pred, labels_1)
    mpa = mean_pixel_accuracy(pred, labels_1, num_classes)
    ious = intersection_over_union(pred, labels_1, num_classes)
    miou = mean_intersection_over_union(pred, labels_1, num_classes)
    fwiou = frequency_weighted_intersection_over_union(pred, labels_1, num_classes)

    print(f"Pixel Accuracy (PA): {pa:.4f}")
    print(f"Mean Pixel Accuracy (MPA): {mpa:.4f}")
    print(f"Intersection over Union (IoU) per class: {ious}")
    print(f"Mean Intersection over Union (mIoU): {miou:.4f}")
    print(f"Frequency Weighted Intersection over Union (FWIoU): {fwiou:.4f}")

dataset_train = GAPartNetPair(
                    Path(root_dir)  / "pth",
                    Path(root_dir)  / "meta",
                    shuffle=True,
                    max_points=20000,
                    augmentation=True,
                    voxelization=True, 
                    group_size = 2,
                    voxel_size=[0.01,0.01,0.01],
                    few_shot = False,
                    few_shot_num=None,
                    pos_jitter = 0.1,
                    color_jitter = 0.3,
                    flip_prob = 0.3,
                    rotate_prob = 0.3,
                )
dataloader_train = DataLoader(
                    dataset_train,
                    batch_size=32,
                    shuffle=True,
                    num_workers=8,
                    collate_fn=data_utils.trivial_batch_collator,
                    pin_memory=True,
                    drop_last=False,
                )

# based blocks
class ResBlock(spconv.SparseModule):
    def __init__(
        self, in_channels: int, out_channels: int, norm_fn: nn.Module, indice_key=None
    ):
        super().__init__()

        if in_channels == out_channels:
            self.shortcut = nn.Identity() # channel 相同就是 x 
        else:
            # assert False
            self.shortcut = spconv.SparseSequential( # feature 层面的全连接
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, \
                bias=False),
                norm_fn(out_channels),
            )

        self.conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

        self.conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(
                out_channels, out_channels, kernel_size=3,
                padding=1, bias=False, indice_key=indice_key,
            ),
            norm_fn(out_channels),
        )

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = x.replace_feature(F.relu(x.features)) # 相当于ReLU

        x = self.conv2(x)
        x = x.replace_feature(F.relu(x.features + shortcut.features))

        return x

class UBlock(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1, # 递归计数器
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks) # 同层次几层

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            ) # 这也能递归？？！

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0] * 2, channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x) # 平层过几次
        shortcut = x

        if len(self.channels) > 1: # 返回条件

            x = self.downsample(x)
            x = self.ublock(x) # 这也能递归？不愧是北大！艺术
            x = self.upsample(x)

            x = x.replace_feature(torch.cat([x.features, shortcut.features],\
                 dim=-1)) # shortcut
            x = self.decoder_blocks(x) # 每层都有decoder_blocks, 因为cut了，所以feature * 2

        return x
    
class SparseUNet(nn.Module):
    def __init__(self, stem: nn.Module, ublock: UBlock):
        super().__init__()

        self.stem = stem
        self.ublock = ublock # 掉了一层壳子

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod # classmethod是个python特殊的方法
    def build( # 相当于另一个构造函数
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3, # 把inchannel和channel对应上
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential( # 通道一样就不管
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)

class UBlock_NoSkip(nn.Module):
    def __init__(
        self,
        channels: List[int],
        block_fn: nn.Module,
        block_repeat: int,
        norm_fn: nn.Module,
        indice_key_id: int = 1,
    ):
        super().__init__()

        self.channels = channels

        encoder_blocks = [
            block_fn(
                channels[0], channels[0], norm_fn, indice_key=f"subm{indice_key_id}"
            )
            for _ in range(block_repeat)
        ]
        self.encoder_blocks = spconv.SparseSequential(*encoder_blocks)

        if len(channels) > 1:
            self.downsample = spconv.SparseSequential(
                spconv.SparseConv3d(
                    channels[0], channels[1], kernel_size=2, stride=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[1]),
                nn.ReLU(),
            )

            self.ublock = UBlock(
                channels[1:], block_fn, block_repeat, norm_fn, indice_key_id + 1
            )

            self.upsample = spconv.SparseSequential(
                spconv.SparseInverseConv3d(
                    channels[1], channels[0], kernel_size=2,
                    bias=False, indice_key=f"spconv{indice_key_id}",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )

            decoder_blocks = [
                block_fn(
                    channels[0], channels[0], norm_fn,
                    indice_key=f"subm{indice_key_id}",
                ),
            ]
            for _ in range(block_repeat -1):
                decoder_blocks.append(
                    block_fn(
                        channels[0], channels[0], norm_fn,
                        indice_key=f"subm{indice_key_id}",
                    )
                )
            self.decoder_blocks = spconv.SparseSequential(*decoder_blocks)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        x = self.encoder_blocks(x)
        # shortcut = x

        if len(self.channels) > 1:
            x = self.downsample(x)
            x = self.ublock(x)
            x = self.upsample(x)

            # x = x.replace_feature(torch.cat([x.features, shortcut.features],\
            #      dim=-1)) # 注释几行话而已
            x = self.decoder_blocks(x)

        return x

class SparseUNet_NoSkip(nn.Module): # 同理注释
    def __init__(self, stem: nn.Module, ublock: UBlock_NoSkip):
        super().__init__()

        self.stem = stem
        self.ublock = ublock

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        x = self.ublock(x)
        return x

    @classmethod
    def build(
        cls,
        in_channels: int,
        channels: List[int],
        block_repeat: int,
        norm_fn: nn.Module,
        without_stem: bool = False,
    ):
        if not without_stem:
            stem = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, channels[0], kernel_size=3,
                    padding=1, bias=False, indice_key="subm1",
                ),
                norm_fn(channels[0]),
                nn.ReLU(),
            )
        else:
            stem = spconv.SparseSequential(
                norm_fn(channels[0]),
                nn.ReLU(),
            )

        block = UBlock(channels, ResBlock, block_repeat, norm_fn, \
            indice_key_id=1)

        return SparseUNet(stem, block)

class STN3d(nn.Module):
    def __init__(self, channel): # channel 看上去应该默认为3
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0] # (bs, features, points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x))) # 一维卷积，放大features维度层次
        x = torch.max(x, 2, keepdim=True)[0] # 点归并成最大features
        x = x.view(-1, 1024) # 展平 

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x))) # 连接到256层特征
        x = self.fc3(x) # 9层

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1) # (bs, 1, 9) #[1 0 0]
        if x.is_cuda: # is_cuda返回0     [0 1 0]
            iden = iden.cuda() #          [0 0 1]
        x = x + iden
        x = x.view(-1, 3, 3) # 预测的是一个单位阵，加上了一个矩阵
        return x

class STNkd(nn.Module):
    def __init__(self, k=64): # 上升到了k维
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k) # 输出是k * k矩阵
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1) # k维度单位阵
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel) # 3维
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64) # 特征也能变换

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x) # 矩阵
        x = x.transpose(2, 1) # 交换 D, N，为了矩阵乘法
        if D > 3: # 分割 features
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans) # x 位置进行变换
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1) # 变回来
        x = F.relu(self.bn1(self.conv1(x))) # 增广D

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat) # 变换features
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x # shortcut
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0] # 增广，features取N上面的最大
        x = x.view(-1, 1024) # 展平
        if self.global_feat:
            return x, trans, trans_feat # 返回的本质是1024feature和
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N) # (bs, 1024, N) N个是一样的
            return torch.cat([x, pointfeat], 1), trans, trans_feat # 决定是否concat，增广是为了concat

class PointNetSegBackbone(nn.Module):
    def __init__(self, pc_dim, fea_dim):
        super(PointNetSegBackbone, self).__init__()
        self.fea_dim = fea_dim
        self.feat = PointNetEncoder(global_feat=False, feature_transform=True, channel=3+pc_dim)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1) # 1024 + 64 feature位置
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.fea_dim, 1) # 干到输出的features
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x) # 给feature降维 
        fea = x.transpose(2,1).contiguous() # D, N 换位
        return fea
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        # return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight = weight) # ?
        mat_diff_loss = feature_transform_reguliarzer(trans_feat) # 正交损失
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale # 你也没返回loss啊

class PointNetBackbone(nn.Module): # 这个就是把pointnet包调出来
    def __init__(
        self,
        pc_dim: int,
        feature_dim: int,
    ):
        super().__init__()
        self.pc_dim = pc_dim
        self.feature_dim = feature_dim
        self.backbone = PointNetSegBackbone(self.pc_dim,self.feature_dim)
    
    def forward(self, input_pc):
        others = {}
        return self.backbone(input_pc), others
    
class Seg_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparseunet = SparseUNet.build(6, [16, 32, 48, 64, 80, 96, 112], 2, functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1))
        self.sem_seg_head = nn.Linear(16, 2)

    def forward(self, pc_batch_1: PointCloudBatch, pc_batch_2: PointCloudBatch):
        voxel_tensor_1 = pc_batch_1.voxel_tensor
        pc_voxel_id_1 = pc_batch_1.pc_voxel_id
        voxel_features = self.sparseunet(voxel_tensor_1)
        pc_feature_1 = voxel_features.features[pc_voxel_id_1]

        voxel_tensor_2 = pc_batch_2.voxel_tensor
        pc_voxel_id_2 = pc_batch_2.pc_voxel_id
        voxel_features = self.sparseunet(voxel_tensor_2)
        pc_feature_2 = voxel_features.features[pc_voxel_id_2]

        seg_1 = self.sem_seg_head(pc_feature_1)
        seg_2 = self.sem_seg_head(pc_feature_2)

        return seg_1, seg_2


def train(model: Seg_test, dataloader_train: DataLoader, epoch: int = 5, lr: float = 0.001, device: torch.device = "cuda:0", save_dir: str = "./log_dir/seg_test/"):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    torch.save(model.state_dict(), save_dir + f"Epoch_[{0}|{epoch}]_Batch_[{0}]_Loss_None.pth") # test code
    for e in range(epoch):
        running_loss = 0.0
        for i, inputs in enumerate(dataloader_train):
            inputs = [inp.to(device) for inp in inputs]
            
            # 从inputs中提取两个点云列表，并进行collate操作
            pc_b_1 = [pc_pair.pc1 for pc_pair in inputs]
            pc_b_2 = [pc_pair.pc2 for pc_pair in inputs]
            
            pc_batch_1 = PointCloud.collate(pc_b_1)
            pc_batch_2 = PointCloud.collate(pc_b_2)

            optimizer.zero_grad()

            seg_1, seg_2 = model(pc_batch_1, pc_batch_2)
            
            labels_1 = pc_batch_1.sem_labels.to(device)
            labels_2 = pc_batch_2.sem_labels.to(device)
            
            labels_1[labels_1 > 0] = 1
            labels_2[labels_2 > 0] = 1
            
            loss_1 = criterion(seg_1, labels_1)
            loss_2 = criterion(seg_2, labels_2)
            
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:  # 每10个batch打印一次损失
                print(f"Epoch [{e + 1}/{epoch}], Batch [{i + 1}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0
        if e % 10 == 0:
            torch.save(model.state_dict(), save_dir + f"Epoch[{e + 1}|{epoch}]_Batch[{i + 1}]_Loss:{running_loss / 10:.4f}.pth")
                

        print(f"Epoch [{e + 1}/{epoch}] completed.")

    print("Training finished.")

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--lr", default=0.001, type=str, help="learning_rate")
    arg.add_argument("--epoch", default=100, type=int, help="epoch")
    arg.add_argument("--path", default="./log_dir/seg_test/", type=str, help="save path")
    conf = arg.parse_args()
    model = Seg_test().cuda()
    train(model, dataloader_train, epoch=conf.epoch, lr=conf.lr, save_dir=conf.path)
    torch.save(model.state_dict(), conf.path + f"/seg_test_SpuNet_{conf.epoch}_epoch_end.pth")
