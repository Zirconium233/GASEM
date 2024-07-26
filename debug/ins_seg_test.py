import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
from network.gap_layers import *
from datasets.datasets_pair import *
import functools
from network.sym_v1 import *
from network.flownet3d import *
from network.gpv_layers import *
from loss.utils import *
from visu.utils import *
from network.utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from datasets.GAPartNet.misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix
from datasets.GAPartNet.dataset.instances import Instances
from epic_ops.reduce import segmented_maxpool
from network.gap_grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from loss.utils import focal_loss, dice_loss, pixel_accuracy, mean_iou

root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
test_intra_dir = "/16T/zhangran/GAPartNet_re_rendered/test_intra"
test_inter_dir = "/16T/zhangran/GAPartNet_re_rendered/test_inter"
def get_datasets(root_dir, test_intra_dir, test_inter_dir, max_points=2000, voxelization=False, shot=False, choose_category: List=None, augmentation=True):
    if shot:
        few_shot = True
        few_shot_num = 20
    else:
        few_shot = False
        few_shot_num = None

    dataset_train = GAPartNetPair(
        Path(root_dir) / "pth",
        Path(root_dir) / "meta",
        shuffle=True,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
    )

    dataset_test_intra = GAPartNetPair(
        Path(test_intra_dir) / "pth",
        Path(test_intra_dir) / "meta",
        shuffle=False,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
    )

    dataset_test_inter = GAPartNetPair(
        Path(test_inter_dir) / "pth",
        Path(test_inter_dir) / "meta",
        shuffle=False,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
    )

    return dataset_train, dataset_test_intra, dataset_test_inter

def get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, batch_size=16, num_workers=8):
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False
    )
    # test_intra_sampler = DistributedSampler(dataset_test_intra, shuffle=False)
    dataloader_test_intra = DataLoader(
        dataset_test_intra,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False,
        # sampler=test_intra_sampler
    )
    # test_inter_sampler = DistributedSampler(dataset_test_inter, shuffle=False)
    dataloader_test_inter = DataLoader(
        dataset_test_inter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False,
        # sampler=test_inter_sampler
    )
    return dataloader_train, dataloader_test_intra, dataloader_test_inter

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
                vertices: "tensor (bs, vetice_num, 3)",# type: ignore
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
        # rotation
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

class test_GPV(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PoseNet9D_Only_R(feat_c_R=1280)
    
    def forward(self, pc_list: List[PointCloudPair]):
        points1 = torch.cat([pc.pc1.points.unsqueeze(0) for pc in pc_list], dim=0)  # pc_list is batch size
        points2 = torch.cat([pc.pc2.points.unsqueeze(0) for pc in pc_list], dim=0)
        p_green_R1, p_red_R1, f_green_R1, f_red_R1 = self.backbone(points1[:,:, 0:3])
        p_green_R2, p_red_R2, f_green_R2, f_red_R2 = self.backbone(points2[:,:, 0:3])
        return (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2)
    


def voxelization_points(
    points: torch.Tensor, voxel_size: Tuple[float, float, float]
):
    bs = points.shape[0]
    voxel_features = []
    voxel_coords = []
    pc_voxel_ids = []
    voxel_coords_ranges = []
    for i in range(bs):
        num_points = points.shape[1]
        pt_xyz = points[i, :, :3]
        points_range_min = pt_xyz.min(0)[0] - 1e-4
        points_range_max = pt_xyz.max(0)[0] + 1e-4
        voxel_feature, voxel_coord, _, pc_voxel_id = voxelize(
            pt_xyz, points[i],
            batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64, device = pt_xyz.device),
            voxel_size=torch.as_tensor(voxel_size, device = pt_xyz.device),
            points_range_min=torch.as_tensor(points_range_min, device = pt_xyz.device),
            points_range_max=torch.as_tensor(points_range_max, device = pt_xyz.device),
            reduction="mean",
        )
        voxel_features.append(voxel_feature)
        voxel_coords.append(voxel_coord)
        pc_voxel_ids.append(pc_voxel_id)
        voxel_coords_range = (voxel_coord.max(0)[0] + 1).clamp(min=128, max=None)
        voxel_coords_ranges.append(voxel_coords_range)
        assert (pc_voxel_id >= 0).all()

    # voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)
    # voxel_coords_range = voxel_coords_range.tolist()
    batch_indices = torch.cat([
        torch.full((point.shape[0],), i, dtype=torch.int32, device="cuda:0")
        for i, point in enumerate(points)
    ], dim=0)
    voxel_batch_indices = torch.cat([
        torch.full((
            voxel_coord.shape[0],), i, dtype=torch.int32, device="cuda:0"
        )
        for i, voxel_coord in enumerate(voxel_coords)
    ], dim=0)
    voxel_features = torch.cat(voxel_features, dim=0)
    voxel_coords = torch.cat(voxel_coords, dim=0)
    voxel_coords = torch.cat([
        voxel_batch_indices[:, None], voxel_coords
    ], dim=-1)
    pc_voxel_id = torch.cat(pc_voxel_ids, dim=0)
    voxel_coords_range = torch.max(torch.stack(voxel_coords_ranges), dim=0)[0]
    voxel_tensor = spconv.SparseConvTensor(
        voxel_features, voxel_coords,
        spatial_shape=voxel_coords_range.tolist(),
        batch_size=bs
    )
    return voxel_tensor, pc_voxel_id, batch_indices

class InsSegTest(nn.Module):
    def __init__(self, backbone_type: str = "SparseUNet", num_part_classes = 10, channels = [16, 32, 48, 64, 80, 96, 112], block_repeat = 2):
        super(InsSegTest, self).__init__()
        self.num_part_classes = num_part_classes
        self.backbone_type = backbone_type
        self.ball_query_radius = 0.12
        self.max_num_points_per_query = 100
        self.min_num_points_per_proposal = 3 # 50 for scannet?
        self.max_num_points_per_query_shift = 500
        self.score_fullscale = 28
        self.score_scale = 50
        if backbone_type == "SparseUNet":
            self.backbone = SparseUNet.build(6, channels, block_repeat, functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1))
            fea_dim = 16
        else:
            raise NotImplementedError("backbone not implemented")
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        self.sem_seg_head = nn.Linear(16, self.num_part_classes)
        self.offset_head = nn.Sequential(
        nn.Linear(fea_dim, fea_dim),
            norm_fn(fea_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fea_dim, 3),
        )
        self.score_unet = SparseUNet.build(
            fea_dim, channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.score_head = nn.Linear(fea_dim, self.num_part_classes - 1)
        self.npcs_unet = SparseUNet.build(
            fea_dim, channels[:2], block_repeat, norm_fn, without_stem=True
        )
        self.npcs_head = nn.Linear(fea_dim, 3 * (self.num_part_classes - 1))
        
        (
            symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3
        ) = get_symmetry_matrix()
        self.symmetry_matrix_1 = symmetry_matrix_1
        self.symmetry_matrix_2 = symmetry_matrix_2
        self.symmetry_matrix_3 = symmetry_matrix_3
    
    def forward(self, points, flow_data, instance_labels = None): # (bs,2048,3) + (bs,3,2048)
        # 假设flow_data是backbone处理后的光流数据
        flow_data = flow_data.permute(0, 2, 1)
        inputs = torch.cat([points, flow_data], dim=2)
        pc_features, batch_indices = self.forward_backbone(inputs)
        cat_inputs = torch.cat([points[i] for i in range(points.shape[0])], dim=0)
        pt_xyz = cat_inputs[:, :3]
        sem_logits = self.sem_seg_head(pc_features)
        sem_preds = torch.argmax(sem_logits.detach(), dim=-1)
        offsets_preds = self.forward_offset(pc_features)
        voxel_tensor, pc_voxel_id, proposals = self.proposal_clustering_and_revoxelize(
            pt_xyz = pt_xyz,
            batch_indices=batch_indices,
            pt_features=pc_features,
            sem_preds=sem_preds,
            offset_preds=offsets_preds,
            instance_labels=instance_labels,
        )
        if proposals is None:
            return sem_preds, sem_logits, offsets_preds, None, None, None
        
        score_logits = self.forward_proposal_score(
            voxel_tensor, pc_voxel_id, proposals
        )
        proposal_offsets_begin = proposals.proposal_offsets[:-1].long()
        score_logits = score_logits.gather(
            1, proposals.sem_preds[proposal_offsets_begin].long()[:, None] - 1
        ).squeeze(1)
        proposals.score_preds = score_logits.detach().sigmoid()
        npcs_logits = self.forward_proposal_npcs(
            voxel_tensor, pc_voxel_id
        )
        return sem_preds, sem_logits, offsets_preds, proposals, score_logits, npcs_logits
    
    def forward_backbone(self, inputs):
        if self.backbone_type == "SparseUNet":
            voxel_tensor, pc_voxel_id, batch_indices = voxelization_points(inputs, [0.01,0.01,0.01])
            voxel_features = self.backbone(voxel_tensor)
            pc_feature = voxel_features.features[pc_voxel_id]
            return pc_feature, batch_indices
        else:
            raise NotImplementedError("backbone not implemented")
    
    def forward_sem_seg(
        self,
        pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        sem_logits = self.sem_seg_head(pc_feature)

        return sem_logits
    
    def forward_offset(
        self,
        pc_feature: torch.Tensor,
    ) -> torch.Tensor:
        offset = self.offset_head(pc_feature)

        return offset
    
    def forward_proposal_score(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
        proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1] # type: ignore
        proposal_offsets_end = proposal_offsets[1:] # type: ignore

        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits
    
    def forward_proposal_npcs(
        self,
        voxel_tensor: spconv.SparseConvTensor,
        pc_voxel_id: torch.Tensor,
    ) -> torch.Tensor:
        npcs_features = self.npcs_unet(voxel_tensor)
        npcs_logits = self.npcs_head(npcs_features.features)
        npcs_logits = npcs_logits[pc_voxel_id]

        return npcs_logits
    

    def proposal_clustering_and_revoxelize(
        self,
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        pt_features: torch.Tensor,
        sem_preds: torch.Tensor,
        offset_preds: torch.Tensor,
        instance_labels: Optional[torch.Tensor],
    ):
        device = pt_xyz.device
        
        if instance_labels is not None:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = sem_preds > 0
        
        pt_xyz = pt_xyz[valid_mask]
        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        offset_preds = offset_preds[valid_mask]
        if instance_labels is not None:
            instance_labels = instance_labels[valid_mask]
            
        # get batch offsets (csr) from batch indices
        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)
        
        # cluster proposals: dual set
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )

        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )
        
        # combine clusters
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (
            num_points_per_proposal >= self.min_num_points_per_proposal
        )
        # proposal to point
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        if instance_labels is not None:
            instance_labels = instance_labels[sorted_indices]

        # re-compact the proposal ids
        proposal_indices = proposal_indices[valid_point_mask]
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            proposal_indices, return_inverse=True, return_counts=True
        )
        num_proposals = num_points_per_proposal.shape[0]

        # get proposal batch offsets
        proposal_offsets = torch.zeros(
            num_proposals + 1, dtype=torch.int32, device=device
        )
        proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

        # voxelization
        voxel_features, voxel_coords, pc_voxel_id = segmented_voxelize(
            pt_xyz, pt_features,
            proposal_offsets, proposal_indices,
            num_points_per_proposal,
            self.score_fullscale, self.score_scale,
        )
        voxel_tensor = spconv.SparseConvTensor(
            voxel_features, voxel_coords.int(),
            spatial_shape=[self.score_fullscale] * 3,
            batch_size=num_proposals,
        )
        if not (pc_voxel_id >= 0).all():
            import pdb
            pdb.set_trace()
            
        proposals = Instances(
            valid_mask=valid_mask,
            sorted_indices=sorted_indices,
            pt_xyz=pt_xyz,
            batch_indices=batch_indices,
            proposal_offsets=proposal_offsets,
            proposal_indices=proposal_indices,
            num_points_per_proposal=num_points_per_proposal,
            sem_preds=sem_preds,
            instance_labels=instance_labels,
        )

        return voxel_tensor, pc_voxel_id, proposals

class ins_seg_loss(nn.Module):
    def __init__(self, use_sem_focal_loss: bool = True, 
                 use_sem_dice_loss: bool = True, 
                 ignore_sem_label: int = -100, 
                 symmetry_indices: List[int] = [0, 1, 3, 3, 2, 0, 3, 2, 4, 1],
                 train_schedule = [5, 10],
                 device = "cuda:0"):
        super(ins_seg_loss, self).__init__()
        self.device = device
        self.start_clustering_epoch = train_schedule[0]
        self.start_npcs_epoch = train_schedule[1]
        self.use_sem_focal_loss = use_sem_focal_loss
        self.use_sem_dice_loss = use_sem_dice_loss
        self.ignore_sem_label = ignore_sem_label
        (
            symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3
        ) = get_symmetry_matrix()
        self.symmetry_matrix_1 = symmetry_matrix_1
        self.symmetry_matrix_2 = symmetry_matrix_2
        self.symmetry_matrix_3 = symmetry_matrix_3
        self.symmetry_indices = torch.as_tensor(symmetry_indices, dtype=torch.int64).to(self.device)
        
        
    def loss_sem_seg(
        self,
        sem_logits: torch.Tensor,
        sem_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_sem_focal_loss:
            loss = focal_loss(
                sem_logits, sem_labels,
                alpha=None,
                gamma=2.0,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )
        else:
            loss = F.cross_entropy(
                sem_logits, sem_labels,
                weight=None,
                ignore_index=self.ignore_sem_label,
                reduction="mean",
            )

        if self.use_sem_dice_loss:
            loss += dice_loss(
                sem_logits[:, :, None, None], sem_labels[:, None, None],
            )

        return loss
    
    def loss_offset(
        self,
        offsets: torch.Tensor,
        gt_offsets: torch.Tensor,
        sem_labels: torch.Tensor,
        instance_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        valid_instance_mask = (sem_labels > 0) & (instance_labels >= 0)

        pt_diff = offsets - gt_offsets
        pt_dist = torch.sum(pt_diff.abs(), dim=-1)
        loss_offset_dist = pt_dist[valid_instance_mask].mean()

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=-1)
        gt_offsets = gt_offsets / (gt_offsets_norm[:, None] + 1e-8)

        offsets_norm = torch.norm(offsets, p=2, dim=-1)
        offsets = offsets / (offsets_norm[:, None] + 1e-8)

        dir_diff = -(gt_offsets * offsets).sum(-1)
        loss_offset_dir = dir_diff[valid_instance_mask].mean()

        return loss_offset_dist, loss_offset_dir
    
    def loss_proposal_score(
        self,
        score_logits: torch.Tensor,
        proposals: Instances,
        num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets, # type: ignore
            proposals.instance_labels, # type: ignore
            proposals.batch_indices, # type: ignore
            num_points_per_instance,
        )
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max = ious.max(-1)[0]
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        return F.binary_cross_entropy_with_logits(score_logits, gt_scores)
    
    def loss_proposal_npcs(
        self,
        npcs_logits: torch.Tensor,
        gt_npcs: torch.Tensor,
        proposals: Instances,
    ) -> torch.Tensor:
        sem_preds, sem_labels = proposals.sem_preds, proposals.sem_labels
        proposal_indices = proposals.proposal_indices
        valid_mask = (sem_preds == sem_labels) & (gt_npcs != 0).any(dim=-1)

        npcs_logits = npcs_logits[valid_mask]
        gt_npcs = gt_npcs[valid_mask]
        sem_preds = sem_preds[valid_mask].long()
        sem_labels = sem_labels[valid_mask]
        proposal_indices = proposal_indices[valid_mask]

        npcs_logits = rearrange(npcs_logits, "n (k c) -> n k c", c=3)
        npcs_logits = npcs_logits.gather(
            1, index=repeat(sem_preds - 1, "n -> n one c", one=1, c=3)
        ).squeeze(1)

        proposals.npcs_preds = npcs_logits.detach()
        proposals.gt_npcs = gt_npcs
        proposals.npcs_valid_mask = valid_mask

        loss_npcs = 0

        # import pdb; pdb.set_trace()
        self.symmetry_indices = self.symmetry_indices.to(sem_preds.device)
        self.symmetry_matrix_1 = self.symmetry_matrix_1.to(sem_preds.device)
        self.symmetry_matrix_2 = self.symmetry_matrix_2.to(sem_preds.device)
        self.symmetry_matrix_3 = self.symmetry_matrix_3.to(sem_preds.device)
        # import pdb; pdb.set_trace()
        symmetry_indices = self.symmetry_indices[sem_preds]
        # group #1
        group_1_mask = symmetry_indices < 3
        symmetry_indices_1 = symmetry_indices[group_1_mask]
        if symmetry_indices_1.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_1_mask], gt_npcs[group_1_mask],
                proposal_indices[group_1_mask],
                self.symmetry_matrix_1[symmetry_indices_1]
            )

        # group #2
        group_2_mask = symmetry_indices == 3
        symmetry_indices_2 = symmetry_indices[group_2_mask]
        if symmetry_indices_2.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_2_mask], gt_npcs[group_2_mask],
                proposal_indices[group_2_mask],
                self.symmetry_matrix_2[symmetry_indices_2 - 3]
            )

        # group #3
        group_3_mask = symmetry_indices == 4
        symmetry_indices_3 = symmetry_indices[group_3_mask]
        if symmetry_indices_3.shape[0] > 0:
            loss_npcs += compute_npcs_loss(
                npcs_logits[group_3_mask], gt_npcs[group_3_mask],
                proposal_indices[group_3_mask],
                self.symmetry_matrix_3[symmetry_indices_3 - 4]
            )

        return loss_npcs
    
    def add_labels(self, pc_pairs, sem_preds, proposals):
        batch_size = len(pc_pairs)
        num_instances = [pc.pc1.num_instances for pc in pc_pairs]
        max_num_instances = max(num_instances)
        sem_labels = torch.cat([pc_pair.pc1.sem_labels for pc_pair in pc_pairs], dim=0)
        instance_labels = torch.cat([pc_pair.pc1.instance_labels for pc_pair in pc_pairs], dim=0)
        if proposals is None:
            return sem_labels, instance_labels
        proposals.sem_labels = sem_labels[proposals.valid_mask][
            proposals.sorted_indices
        ]
        proposals.instance_labels = instance_labels[proposals.valid_mask][proposals.sorted_indices]
        proposals.sem_preds = sem_preds[proposals.valid_mask][proposals.sorted_indices]
        num_points_per_instance = torch.zeros(
            batch_size, max_num_instances, dtype=torch.int32, device=self.device
        )
        instance_sem_labels = torch.full(
            (batch_size, max_num_instances), -1, dtype=torch.int32, device=self.device
        )
        for i, pc in enumerate(pc_pairs):
            num_points_per_instance[i, :pc.pc1.num_instances] = pc.pc1.num_points_per_instance 
            instance_sem_labels[i, :pc.pc1.num_instances] = pc.pc1.instance_sem_labels 
        proposals.num_points_per_instance = num_points_per_instance
        proposals.instance_sem_labels = instance_sem_labels

        return sem_labels, instance_labels
    
    def forward(self, epoch, sem_logits, sem_labels, instance_labels, offsets_preds, proposals, score_logits, npcs_logits, pc_pairs):
        # self.add_labels(pc_pairs, sem_preds, proposals)
        pt_xyz = torch.cat([pc_pair.pc1.points[:, :3] for pc_pair in pc_pairs], dim=0)
        instance_regions = torch.cat([pc_pair.pc1.instance_regions for pc_pair in pc_pairs], dim=0)
        gt_offsets = instance_regions[:, :3] - pt_xyz
        gt_npcs = torch.cat([pc_pair.pc1.gt_npcs for pc_pair in pc_pairs], dim=0)
        loss_sem = self.loss_sem_seg(sem_logits, sem_labels)
        loss_offset_dist, loss_offset_dir = self.loss_offset(offsets_preds, gt_offsets, sem_labels, instance_labels)
        if proposals is None:
            return loss_sem + loss_offset_dist + loss_offset_dir
        loss_score = self.loss_proposal_score(score_logits, proposals, proposals.num_points_per_instance)
        gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
        loss_npcs = self.loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
        if epoch < self.start_clustering_epoch:
            return loss_sem + loss_offset_dist + loss_offset_dir
        elif epoch < self.start_npcs_epoch:
            return loss_sem + loss_offset_dist + loss_offset_dir + loss_score
        else:
            return loss_sem + loss_offset_dist + loss_offset_dir + loss_score + loss_npcs

def train_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, dataloader_test_intra, dataloader_test_inter, num_epochs, lr, train_schedule, log_dir, device):
    optimizer = torch.optim.Adam(ins_seg.parameters(), lr=lr)
    ins_seg.train()
    gpv_net.eval()
    flownet.eval() # fix flownet parameters
    ins_seg = ins_seg.to(device)
    gpv_net = gpv_net.to(device)
    flownet = flownet.to(device)
    criterion = ins_seg_loss(train_schedule=train_schedule)
    log_dir = log_dir + "/" + str(datetime.today())
    writer = SummaryWriter(log_dir=log_dir)
    print("_________________________train_epoch___________________________")
    global_step = 0
    for epoch in range(num_epochs):
        total_loss = 0
        total_all_accu = 0
        total_pixel_accu = 0
        if epoch == 0:
            print("______________________first_test_epoch_________________________")
            torch.save(ins_seg.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', criterion)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', criterion)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, device, writer, epoch, 'test_train', criterion)
        for batch_idx, batch in enumerate(dataloader_train):
            bs = len(batch)
            pc_pairs = [pair.to(device) for pair in batch]
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)            
            # Convert predicted vectors and ground truth vectors back to rotation matrices
            pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
            pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
            input1 = torch.stack([(pred_rot_matrices1[i].transpose(0,1) @ pc_pairs[i].pc1.points[0:2048,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
            feat1 = torch.stack([pc_pair.pc1.points[0:2048,3:6] for pc_pair in pc_pairs], dim=0)
            input2 = torch.stack([(pred_rot_matrices2[i].transpose(0,1) @ pc_pairs[i].pc2.points[0:2048,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
            feat2 = torch.stack([pc_pair.pc2.points[0:2048,3:6] for pc_pair in pc_pairs], dim=0)
            flow_data = flownet(
                input1.transpose(1,2).contiguous(),
                input2.transpose(1,2).contiguous(),
                feat1.transpose(1,2).contiguous(),
                feat2.transpose(1,2).contiguous()
            )
            pt_xyz = torch.cat([pc_pair.pc1.points[:,0:3].unsqueeze(0) for pc_pair in pc_pairs], dim=0)
            instance_labels = torch.cat([pc_pair.pc1.instance_labels for pc_pair in pc_pairs], dim=0)
            # flow_data = torch.cat([pc_pair.pc1.points[:,3:6].unsqueeze(0) for pc_pair in pc_pairs], dim=0).permute(0,2,1)
            # points = torch.cat([pt_xyz, flow_data.permute(0, 2, 1)], dim=2)
            # cat_inputs = torch.cat([points[i] for i in range(points.shape[0])], dim=0)
            sem_preds, sem_logits, offsets_preds, proposals, score_logits, npcs_logits = ins_seg(pt_xyz, flow_data, instance_labels)
            sem_labels, instance_labels = criterion.add_labels(pc_pairs, sem_preds, proposals)
            all_accu = (sem_preds == sem_labels).sum().float() / (sem_labels.shape[0])
            instance_mask = sem_labels > 0
            pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
            # criterion.add_labels(pc_pairs, sem_preds, proposals)
            loss = criterion(epoch, sem_logits, sem_labels, instance_labels, offsets_preds, proposals, score_logits, npcs_logits, pc_pairs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            total_loss += loss.item()
            total_all_accu += all_accu.item()
            total_pixel_accu += pixel_accu
            # 每10个batch记录一次loss
            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/all_accu', all_accu.item(), global_step)
                writer.add_scalar('train/pixel_accu', pixel_accu, global_step)
                print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")

        avg_loss = total_loss / len(dataloader_train)
        avg_all_accu = total_all_accu / len(dataloader_train)
        avg_pixel_accu = total_pixel_accu / len(dataloader_train)
        print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)
        writer.add_scalar('train/avg_all_accu', avg_all_accu * 100, epoch)
        writer.add_scalar('train/avg_pixel_accu', avg_pixel_accu * 100, epoch)
        if (epoch + 1) % 10 == 0:
            torch.save(ins_seg.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', criterion)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', criterion)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, device, writer, epoch, 'test_train', criterion)

def test_ins_seg(ins_seg, gpv_net, flownet, dataloader, device, writer, epoch, phase, criterion):
    print("______________________" + phase + "_______________________")
    ins_seg.eval()
    gpv_net.eval()
    flownet.eval()
    all_sem_preds = []
    all_sem_labels = []
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    all_proposals = []
    with torch.no_grad():
        total_loss = 0
        total_all_accu = 0
        total_pixel_accu = 0
        for batch in tqdm(dataloader):
            bs = len(batch)
            pc_pairs = [pair.to(device) for pair in batch]
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)
            pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
            pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
            all_pred_rot_matrices.append(pred_rot_matrices1.cpu())
            all_pred_rot_matrices.append(pred_rot_matrices2.cpu())
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2.T for pc in pc_pairs]))
            gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1, True)
            gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2, True)
            all_gt_rot_matrices.append(gt_rot_matrices1.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices2.cpu())
            input1 = torch.stack([(pred_rot_matrices1[i].transpose(0,1) @ pc_pairs[i].pc1.points[0:2048,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
            feat1 = torch.stack([pc_pair.pc1.points[0:2048,3:6] for pc_pair in pc_pairs], dim=0)
            input2 = torch.stack([(pred_rot_matrices2[i].transpose(0,1) @ pc_pairs[i].pc2.points[0:2048,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
            feat2 = torch.stack([pc_pair.pc2.points[0:2048,3:6] for pc_pair in pc_pairs], dim=0)
            flow_data = flownet(
                input1.transpose(1,2).contiguous(),
                input2.transpose(1,2).contiguous(),
                feat1.transpose(1,2).contiguous(),
                feat2.transpose(1,2).contiguous()
            )
            pt_xyz = torch.cat([pc_pair.pc1.points[:,0:3].unsqueeze(0) for pc_pair in pc_pairs], dim=0)
            instance_labels = torch.cat([pc_pair.pc1.instance_labels for pc_pair in pc_pairs], dim=0)
            # flow_data = torch.cat([pc_pair.pc1.points[:,3:6].unsqueeze(0) for pc_pair in pc_pairs], dim=0).permute(0,2,1)
            # points = torch.cat([pt_xyz, flow_data.permute(0, 2, 1)], dim=2)
            # cat_inputs = torch.cat([points[i] for i in range(points.shape[0])], dim=0)
            sem_preds, sem_logits, offsets_preds, proposals, score_logits, npcs_logits = ins_seg(pt_xyz, flow_data, instance_labels)
            sem_labels, instance_labels = criterion.add_labels(pc_pairs, sem_preds, proposals)
            loss = criterion(epoch, sem_logits, sem_labels, instance_labels, offsets_preds, proposals, score_logits, npcs_logits, pc_pairs)
            all_accu = (sem_preds == sem_labels).sum().float() / (sem_labels.shape[0])
            instance_mask = sem_labels > 0
            pixel_accu = pixel_accuracy(sem_preds[instance_mask], sem_labels[instance_mask])
            total_loss += loss.item()
            total_all_accu += all_accu.item()
            total_pixel_accu += pixel_accu
            all_sem_preds.append(sem_preds)
            all_sem_labels.append(sem_labels)
            if proposals is not None:
                # proposals = filter_invalid_proposals(
                #     proposals,
                #     score_threshold=0.09,
                #     min_num_points_per_proposal=3
                # )
                # proposals = apply_nms(proposals, 0.3)
                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
                all_proposals.append(proposals)
    all_sem_preds = torch.cat(all_sem_preds, dim=0)
    all_sem_labels = torch.cat(all_sem_labels, dim=0)
    miou = mean_iou(all_sem_preds, all_sem_labels, num_classes=10)
    thes = [0.5 + 0.05 * i for i in range(10)]
    aps = []
    for the in thes:
        if len(all_proposals) != 0:
            ap = compute_ap(all_proposals, 10, the)
        else:
            ap = None
        if ap is not None:
            aps.append(ap)
        if the == 0.5:
            ap50 = ap
    mAP = np.array(aps)
    # 消除nan后再评价
    mAP = np.nanmean(mAP) if len(aps) != 0 else 0
    all_pred_rot_matrices = torch.cat(all_pred_rot_matrices, dim=0)
    all_gt_rot_matrices = torch.cat(all_gt_rot_matrices, dim=0)
    mean_rot_error = calculate_pose_metrics(
        all_pred_rot_matrices, all_gt_rot_matrices
    )
    mean_all_accu = total_all_accu / len(dataloader)
    mean_pixel_accu = total_pixel_accu / len(dataloader)
    # print result
    print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean AP@50: {np.nanmean(ap50) * 100 if ap50 is not None else 0:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mAP: {mAP * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mIoU: {miou * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean All Accu: {mean_all_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean Pixel Accu: {mean_pixel_accu * 100:.4f}")
    # record results
    if writer is not None:
        writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
        writer.add_scalar(
            f"{phase}/mean_AP@50",
            np.nanmean(ap50) * 100 if ap50 is not None else 0,
            epoch
        )
        writer.add_scalar(
            f"{phase}/mAP",
            mAP * 100,
            epoch
        )
        writer.add_scalar(
            f"{phase}/mIoU",
            miou * 100,
            epoch
        )
        writer.add_scalar(
            f"{phase}/all_accu",
            mean_all_accu * 100,
            epoch
        )
        writer.add_scalar(
            f"{phase}/all_accu",
            mean_pixel_accu * 100,
            epoch
        )
        for class_idx in range(1, 10):
            partname = PART_ID2NAME[class_idx]
            writer.add_scalar(
                f"{phase}/AP@50_{partname}",
                np.nanmean(ap50[class_idx - 1]) * 100 if ap50 is not None else 0,
                epoch
            )
    ins_seg.train()

        

        
if __name__ == "__main__":

    gpv_net = test_GPV().cuda()
    gpv_net.load_state_dict(torch.load("log_dir/GPV_test_new_loss/2024-06-09 13:38:03.887472/GPV_[100|100].pth"))
    gpv_net.eval()

    flownet = FlowNet3D().cuda()
    flownet.load_state_dict(torch.load("/home/zhangran/desktop/GithubClone/flownet3d_pytorch/pretrained_model/model.best.t7"))
    flownet.eval()

    ins_seg = InsSegTest().cuda()
    ins_seg.eval()

    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=False, shot=False, choose_category=None, max_points=2048, augmentation=False)
    dataloader_train, dataloader_test_intra, dataloader_test_inter = get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, num_workers=0, batch_size=8)

    train_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, dataloader_test_intra, dataloader_test_inter, 100, 0.001, [5, 10], "log_dir/ins_seg_test", "cuda:0")