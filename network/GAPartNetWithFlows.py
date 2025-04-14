import copy
import torch
import numpy as np
import functools
import torch.nn as nn
import spconv.pytorch as spconv
import torch.nn.functional as F
from einops import rearrange, repeat
from network.gap_layers import SparseUNet
from network.flownet3d import FlowNet3D
from datasets.GAPartNet.dataset.point_cloud import PointCloud, PointCloudBatch
from typing import Optional, List, Tuple
from datasets.GAPartNet.dataset.instances import Instances
from datasets.GAPartNet.misc.pose_fitting import estimate_similarity_transform
from datasets.GAPartNet.misc.info import get_symmetry_matrix, SYMMETRY_MATRIX
from datasets.datasets_pair import apply_voxelization

symmetry_indices = [0, 1, 3, 3, 2, 0, 3, 2, 4, 1]

def get_symmetry_matrices_from_sem_label(sem_label):
    symmetry_index = symmetry_indices[sem_label]
    symmetry_matrices = SYMMETRY_MATRIX[symmetry_index]
    symmetry_matrices = torch.tensor(symmetry_matrices).float().cuda()
    return symmetry_matrices

def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
) -> torch.Tensor:
    # if ignore_index is not None:
    #     valid_mask = targets != ignore_index
    #     targets = targets[valid_mask]
    #
    #     if targets.shape[0] == 0:
    #         return torch.tensor(0.0).to(dtype=inputs.dtype, device=inputs.device)
    #
    #     inputs = inputs[valid_mask]

    log_p = F.log_softmax(inputs, dim=-1)
    ce_loss = F.nll_loss(
        log_p, targets, weight=alpha, ignore_index=ignore_index, reduction="none"
    )
    log_p_t = log_p.gather(1, targets[:, None]).squeeze(-1)
    loss = ce_loss * ((1 - log_p_t.exp()) ** gamma)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def one_hot(
        labels: torch.Tensor,
        num_classes: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        eps: float = 1e-6,
) -> torch.Tensor:
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxNxHxW. Got: {input.shape}")

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(f"input and target shapes must be the same. Got: {input.shape} and {target.shape}")

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)

    return torch.mean(-dice_score + 1.0)


@torch.no_grad()
def segmented_reduce(
        values: torch.Tensor,
        segment_offsets_begin: torch.Tensor,
        segment_offsets_end: torch.Tensor,
        mode: str = "sum",
) -> torch.Tensor:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    if mode == "sum":
        mode_id = 0
    elif mode == "min":
        mode_id = 1
    elif mode == "max":
        mode_id = 2
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return torch.ops.epic_ops.segmented_reduce(
        values, segment_offsets_begin, segment_offsets_end, mode_id
    )


@torch.no_grad()
def expand_csr(
        offsets: torch.Tensor,
        output_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets.contiguous()

    return torch.ops.epic_ops.expand_csr(
        offsets, output_size
    )


def voxelize(
        points: torch.Tensor,
        pt_features: torch.Tensor,
        batch_offsets: torch.Tensor,
        voxel_size: torch.Tensor,
        points_range_min: torch.Tensor,
        points_range_max: torch.Tensor,
        reduction: str = "mean",
        max_points_per_voxel: int = 9223372036854775807,
        max_voxels: int = 9223372036854775807,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_points = points.shape[0]

    # input can be both on cpu and cuda.
    WORKAROUND = points.device.type == 'cuda'
    if WORKAROUND:
        points, batch_offsets, points_range_min, points_range_max = points.cpu(), \
            batch_offsets.cpu(), points_range_min.cpu(), points_range_max.cpu()

    with torch.no_grad():
        (voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits,) = torch.ops.open3d.voxelize(
            points,
            row_splits=batch_offsets,
            voxel_size=voxel_size,
            points_range_min=points_range_min,
            points_range_max=points_range_max,
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels, )

        if WORKAROUND:
            voxel_coords, voxel_point_indices, voxel_point_row_splits, voxel_batch_splits = voxel_coords.cuda(), \
                voxel_point_indices.cuda(), voxel_point_row_splits.cuda(), voxel_batch_splits.cuda()

        batch_indices, _ = expand_csr(voxel_batch_splits, voxel_coords.shape[0])
        voxel_indices, num_points_per_voxel = expand_csr(voxel_point_row_splits, voxel_point_indices.shape[0])

        if voxel_point_indices.shape[0] == num_points:
            pc_voxel_id = torch.empty_like(voxel_point_indices)
        else:
            pc_voxel_id = torch.full(
                (num_points,), -1,
                dtype=voxel_point_indices.dtype, device=voxel_point_indices.device
            )
        pc_voxel_id.scatter_(dim=0, index=voxel_point_indices, src=voxel_indices)

    # pdb.set_trace()
    pt_features = pt_features[voxel_point_indices]
    voxel_features = torch.segment_reduce(pt_features, reduction, lengths=num_points_per_voxel)
    return voxel_features, voxel_coords, batch_indices, pc_voxel_id


def segmented_voxelize(
        pt_xyz: torch.Tensor,
        pt_features: torch.Tensor,
        segment_offsets: torch.Tensor,
        segment_indices: torch.Tensor,
        num_points_per_segment: torch.Tensor,
        score_fullscale: float,
        score_scale: float,
        random_offsets=None
):
    torch.random.manual_seed(233)

    segment_offsets_begin = segment_offsets[:-1]
    segment_offsets_end = segment_offsets[1:]

    segment_coords_mean = segmented_reduce(
        pt_xyz, segment_offsets_begin, segment_offsets_end, mode="sum"
    ) / num_points_per_segment[:, None]

    centered_points = pt_xyz - segment_coords_mean[segment_indices]

    segment_coords_min = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="min"
    )
    segment_coords_max = segmented_reduce(
        centered_points, segment_offsets_begin, segment_offsets_end, mode="max"
    )

    # rescale the proposals into a cube with xx scale. proposals are normalized.
    # score_fullscale = 50.
    # score_scale = 50.
    segment_scales = 1. / (
            (segment_coords_max - segment_coords_min) / score_fullscale
    ).max(-1)[0] - 0.01
    segment_scales = torch.clamp(segment_scales, min=None, max=score_scale)

    min_xyz = segment_coords_min * segment_scales[..., None]
    max_xyz = segment_coords_max * segment_scales[..., None]

    segment_scales = segment_scales[segment_indices]
    scaled_points = centered_points * segment_scales[..., None]

    torch.random.manual_seed(233)
    range_xyz = max_xyz - min_xyz
    if random_offsets is None:
        random_offsets = torch.clamp(
            score_fullscale - range_xyz - 0.001, min=0
        ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device) + torch.clamp(
            score_fullscale - range_xyz + 0.001, max=0
        ) * torch.rand(3, dtype=min_xyz.dtype, device=min_xyz.device)
    offsets = -min_xyz + random_offsets
    scaled_points += offsets[segment_indices]
    score_fullscale = float(score_fullscale)

    # notice that here we have a notion of voxel and a notion of pc.
    voxel_features, voxel_coords, voxel_batch_indices, pc_voxel_id = voxelize(
        scaled_points,
        pt_features,
        batch_offsets=segment_offsets.long(),
        voxel_size=torch.as_tensor([1., 1., 1.], device=scaled_points.device),
        points_range_min=torch.as_tensor([0., 0., 0.], device=scaled_points.device),
        points_range_max=torch.as_tensor([score_fullscale, score_fullscale, score_fullscale],
                                         device=scaled_points.device),
        reduction="mean",
    )
    voxel_coords = torch.cat([voxel_batch_indices[:, None], voxel_coords], dim=1)

    return voxel_features, voxel_coords, pc_voxel_id, random_offsets


def segmented_maxpool(
        values: torch.Tensor,
        segment_offsets_begin: torch.Tensor,
        segment_offsets_end: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    values = values.contiguous()
    segment_offsets_begin = segment_offsets_begin.contiguous()
    segment_offsets_end = segment_offsets_end.contiguous()

    return torch.ops.epic_ops.segmented_maxpool(
        values, segment_offsets_begin, segment_offsets_end
    )


@torch.no_grad()
def connected_components_labeling(
        indices: torch.Tensor,
        edges: torch.Tensor,
        compacted: bool = True,
) -> torch.Tensor:
    indices = indices.contiguous()
    edges = edges.contiguous()

    return torch.ops.epic_ops.connected_components_labeling(
        indices, edges, compacted
    )


@torch.no_grad()
def ball_query(
        points: torch.Tensor,
        query: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
        radius: float,
        num_samples: int,
        point_labels: Optional[torch.Tensor] = None,
        query_labels: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    points = points.contiguous()
    query = query.contiguous()
    batch_indices = batch_indices.contiguous()
    batch_offsets = batch_offsets.contiguous()

    if point_labels is not None:
        point_labels = point_labels.contiguous()

    if query_labels is not None:
        query_labels = query_labels.contiguous()

    return torch.ops.epic_ops.ball_query(
        points, query, batch_indices, batch_offsets, radius, num_samples,
        point_labels, query_labels,
    )


# @torch.jit.script
def cluster_proposals(
        pt_xyz: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
        sem_preds: torch.Tensor,
        ball_query_radius: float,
        max_num_points_per_query: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = pt_xyz.device
    index_dtype = batch_indices.dtype

    # knn ball-query cluster.
    # clustered indices per point, and its associated number of points.
    clustered_indices, num_points_per_query = ball_query(
        pt_xyz,
        pt_xyz,
        batch_indices,
        batch_offsets,
        ball_query_radius,
        max_num_points_per_query,
        point_labels=sem_preds,
        query_labels=sem_preds,
    )

    # assume max-number of points per query, here we have an upper bound of ccl-indices.
    ccl_indices_begin = torch.arange(
        pt_xyz.shape[0], dtype=index_dtype, device=device
    ) * max_num_points_per_query
    # compute the actual end of ccl-indices.
    ccl_indices_end = ccl_indices_begin + num_points_per_query
    # stack them together for later use.
    ccl_indices = torch.stack([ccl_indices_begin, ccl_indices_end], dim=1)
    # have a ccl. here we have something like -
    # tensor([ 0,  1,  2,  3,  4,  3,  0,  7,  0,  9,  0,  0, 12,  0,  3,  0, 16, 17 ]).
    cc_labels = connected_components_labeling(
        ccl_indices.view(-1), clustered_indices.view(-1), compacted=False
    )

    # sort the cc-labels. thus we have something like.
    # tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    sorted_cc_labels, sorted_indices = torch.sort(cc_labels)
    return sorted_cc_labels, sorted_indices


def filter_invalid_proposals(
        proposals: Instances,
        score_threshold: float,
        min_num_points_per_proposal: int,
) -> Instances:
    score_preds = proposals.score_preds
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    valid_proposals_mask = (
                                   score_preds > score_threshold
                           ) & (num_points_per_proposal > min_num_points_per_proposal)
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
    )


def apply_nms(
        proposals: Instances,
        iou_threshold: float = 0.3,
):
    score_preds = proposals.score_preds
    sorted_indices = proposals.sorted_indices
    proposal_offsets = proposals.proposal_offsets
    proposal_indices = proposals.proposal_indices
    num_points_per_proposal = proposals.num_points_per_proposal

    values = torch.ones(
        sorted_indices.shape[0], dtype=torch.float32, device=sorted_indices.device
    )
    csr = torch.sparse_csr_tensor(
        proposal_offsets.int(), sorted_indices.int(), values,
        dtype=torch.float32, device=sorted_indices.device,
    )
    intersection = csr @ csr.t()
    intersection = intersection.to_dense()
    union = num_points_per_proposal[:, None] + num_points_per_proposal[None, :]
    union = union - intersection

    ious = intersection / (union + 1e-8)
    keep = nms(ious.cuda(), score_preds.cuda(), iou_threshold)
    keep = keep.to(score_preds.device)

    valid_proposals_mask = torch.zeros(
        ious.shape[0], dtype=torch.bool, device=score_preds.device
    )
    valid_proposals_mask[keep] = True
    valid_points_mask = valid_proposals_mask[proposal_indices]

    proposal_indices = proposal_indices[valid_points_mask]
    _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_inverse=True, return_counts=True
    )
    num_proposals = num_points_per_proposal.shape[0]

    proposal_offsets = torch.zeros(
        num_proposals + 1, dtype=torch.int32, device=proposal_indices.device
    )
    proposal_offsets[1:] = num_points_per_proposal.cumsum(0)

    if proposals.npcs_valid_mask is not None:
        valid_npcs_mask = valid_points_mask[proposals.npcs_valid_mask]
    else:
        valid_npcs_mask = valid_points_mask

    return Instances(
        valid_mask=proposals.valid_mask,
        sorted_indices=proposals.sorted_indices[valid_points_mask],
        pt_xyz=proposals.pt_xyz[valid_points_mask],
        batch_indices=proposals.batch_indices[valid_points_mask],
        proposal_offsets=proposal_offsets,
        proposal_indices=proposal_indices,
        num_points_per_proposal=num_points_per_proposal,
        sem_preds=proposals.sem_preds[valid_points_mask],
        score_preds=proposals.score_preds[valid_proposals_mask],
        npcs_preds=proposals.npcs_preds[
            valid_npcs_mask
        ] if proposals.npcs_preds is not None else None,
        sem_labels=proposals.sem_labels[
            valid_points_mask
        ] if proposals.sem_labels is not None else None,
        instance_labels=proposals.instance_labels[
            valid_points_mask
        ] if proposals.instance_labels is not None else None,
        instance_sem_labels=proposals.instance_sem_labels,
        num_points_per_instance=proposals.num_points_per_instance,
        gt_npcs=proposals.gt_npcs[
            valid_npcs_mask
        ] if proposals.gt_npcs is not None else None,
        npcs_valid_mask=proposals.npcs_valid_mask[valid_points_mask] \
            if proposals.npcs_valid_mask is not None else None,
        ious=proposals.ious[
            valid_proposals_mask
        ] if proposals.ious is not None else None,
    )


@torch.no_grad()
def nms(
        ious: torch.Tensor,
        scores: torch.Tensor,
        threshold: float,
) -> torch.Tensor:
    ious = ious.contiguous()
    scores = scores.contiguous()

    return torch.ops.epic_ops.nms(ious, scores, threshold)


@torch.no_grad()
def batch_instance_seg_iou(
        proposal_offsets: torch.Tensor,
        instance_labels: torch.Tensor,
        batch_indices: torch.Tensor,
        num_points_per_instance: torch.Tensor,
) -> torch.Tensor:
    proposal_offsets = proposal_offsets.contiguous()
    instance_labels = instance_labels.contiguous()
    batch_indices = batch_indices.contiguous()
    num_points_per_instance = num_points_per_instance.contiguous()

    return torch.ops.epic_ops.batch_instance_seg_iou(
        proposal_offsets, instance_labels, batch_indices, num_points_per_instance
    )


# @torch.jit.script
def get_gt_scores(
        ious: torch.Tensor, fg_thresh: float = 0.75, bg_thresh: float = 0.25
) -> torch.Tensor:
    # if iou > fg_thresh, then it is 1.
    # if iou < bg_thresh, then it is 0.
    # if iou in intermidiate, then it is a function.

    fg_mask = ious > fg_thresh
    bg_mask = ious < bg_thresh
    intermidiate_mask = ~(fg_mask | bg_mask)

    gt_scores = fg_mask.float()
    k = 1 / (fg_thresh - bg_thresh)
    b = bg_thresh / (bg_thresh - fg_thresh)
    gt_scores[intermidiate_mask] = ious[intermidiate_mask] * k + b

    return gt_scores


def compute_npcs_loss(
        npcs_preds: torch.Tensor,
        gt_npcs: torch.Tensor,
        proposal_indices: torch.Tensor,
        symmetry_matrix: torch.Tensor,
) -> torch.Tensor:
    _, num_points_per_proposal = torch.unique_consecutive(
        proposal_indices, return_counts=True
    )

    # gt_npcs: n, 3 -> n, 1, 1, 3
    # symmetry_matrix: n, m, 3, 3
    # symmetry matrix is multiplied from right. 
    gt_npcs = gt_npcs[:, None, None, :] @ symmetry_matrix
    # n, m, 1, 3 -> n, m, 3
    gt_npcs = gt_npcs.squeeze(2)

    # npcs_preds: n, 3 -> n, 1, 3
    # gt-npcs has been added with 0.5 in each axis.
    # this does not affect rotation.
    dist2 = (npcs_preds[:, None, :] - gt_npcs - 0.5) ** 2
    # n, m, 3 -> n, m
    dist2 = dist2.sum(dim=-1)

    # a huber loss.
    loss = torch.where(
        dist2 <= 0.01,
        5 * dist2, torch.sqrt(dist2) - 0.05,
    )
    loss = torch.segment_reduce(
        loss, "mean", lengths=num_points_per_proposal
    )
    loss, _ = loss.min(dim=-1)
    return loss.mean()


class GaPartNetWithFlows(nn.Module):
    """
    Rebuild gapartnet and change the inputs
    and outputs to be compatible with our flow data.
    """

    def __init__(self, args):
        super().__init__()
        # Keep most hyper-parameter same as GAPartNet
        self.args = args
        self.cat_points_and_flows = args.cat_points_and_flows
        self.in_channels = 6 if not args.cat_points_and_flows else 9
        self.cat_features = args.cat_features
        try:
            self.improve_pose = args.improve_pose
        except:
            self.improve_pose = False
        try:
            # test for independent backbones for two point clouds
            self.two_backbones = args.two_backbones
        except:
            self.two_backbones = False
        try:
            # When we fix the camera, we do not need to use flownet during this process
            self.fix_camera = args.fix_camera
        except:
            self.fix_camera = False

        try:
            self.offset_cat = args.offset_cat
        except:
            self.offset_cat = False
        if self.fix_camera:
            self.cat_features = True
        
        if self.cat_features:
            # directly concatenate features of two point clouds (not recommended)
            assert self.cat_points_and_flows == False, "cat_features is not compatible with cat_points_and_flows"
        self.num_part_classes = 10
        self.use_sem_focal_loss = True
        self.use_sem_dice_loss = True
        self.ignore_sem_label = -100

        self.ball_query_radius = 0.04
        self.max_num_points_per_query = 50
        self.min_num_points_per_proposal = 5 
        self.max_num_points_per_query_shift = 300

        self.current_epoch = 0
        self.start_scorenet, self.start_npcs = 5, 10 # train schedule
        self.start_clustering = min(self.start_scorenet, self.start_npcs)

        self.score_fullscale = 28
        self.score_scale = 50

        self.val_score_threshold = 0.09 # When test pose, we matually edit these value
        self.val_min_num_points_per_proposal = 3
        self.val_nms_iou_threshold = 0.3
        self.val_ap_iou_threshold = 0.5

        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        in_channels = self.in_channels
        channels = (16, 32, 48, 64, 80, 96, 112)
        block_repeat = 2
        self.backbone = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        if self.two_backbones:
            self.backbone2 = SparseUNet.build(in_channels, channels, block_repeat, norm_fn)
        self.feature_dim = channels[0]
        self.offset_head = nn.Sequential(
            nn.Linear(self.feature_dim * 2 if self.offset_cat else self.feature_dim, self.feature_dim),
            norm_fn(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 3),
        )
        if self.fix_camera:
            self.sem_seg_unet = SparseUNet.build(
                self.feature_dim, channels[:4], block_repeat, norm_fn, without_stem=self.feature_dim == channels[0]
            )
            self.sem_seg_head = nn.Sequential( # the features is concatenated features, so we need more layers
                nn.Linear(self.feature_dim, self.feature_dim),
                norm_fn(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.num_part_classes),
            )
        elif self.cat_features and not self.fix_camera and not self.improve_pose:
            # add a layer to extract features from two point clouds, here we use flownet
            self.flownet = FlowNet3D(in_channel=16, out_channel=16)
            self.sem_seg_unet = SparseUNet.build(
                self.feature_dim * 2, channels[:4], block_repeat, norm_fn, without_stem=self.feature_dim * 2 == channels[0]
            )
            self.sem_seg_head = nn.Sequential( # the features is concatenated features, so we need more layers
                nn.Linear(self.feature_dim, self.feature_dim),
                norm_fn(self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, self.num_part_classes),
            )
            # self.feature_dim *= 2
            # self.sem_seg_head = nn.Linear(self.feature_dim, self.num_part_classes)
        elif self.cat_features and self.improve_pose:
            self.flownet = FlowNet3D(in_channel=16, out_channel=16)
            self.sem_seg_unet = SparseUNet.build(
                self.feature_dim * 2, channels[1:6], block_repeat, norm_fn, without_stem=self.feature_dim * 2 == channels[1]
            )
            self.sem_seg_head = nn.Sequential(
                nn.Linear(channels[1], channels[0]),
                norm_fn(channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(channels[0], self.num_part_classes),
            )
        else:
            self.sem_seg_head = nn.Linear(self.feature_dim, self.num_part_classes)
        if self.improve_pose and self.fix_camera:
            self.score_unet = SparseUNet.build(
                self.feature_dim, channels[0:3], block_repeat, norm_fn, without_stem=self.feature_dim == channels[0]
            )
            self.score_head = nn.Sequential(
                nn.Linear(channels[0], channels[0]),
                norm_fn(channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(channels[0], self.num_part_classes - 1),
            )
        elif self.improve_pose:
            self.score_unet = SparseUNet.build(
                self.feature_dim * 2, channels[1:4], block_repeat, norm_fn, without_stem=self.feature_dim * 2 == channels[1]
            )
            self.score_head = nn.Sequential(
                nn.Linear(channels[1], channels[0]),
                norm_fn(channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(channels[0], self.num_part_classes - 1),
            )
        else:
            self.score_unet = SparseUNet.build(
                self.feature_dim, channels[:2], block_repeat, norm_fn, without_stem=self.feature_dim == channels[0]
            )
            self.score_head = nn.Linear(channels[0], self.num_part_classes - 1)

        symmetry_matrix_1, symmetry_matrix_2, symmetry_matrix_3 = get_symmetry_matrix()
        self.symmetry_matrix_1 = symmetry_matrix_1
        self.symmetry_matrix_2 = symmetry_matrix_2
        self.symmetry_matrix_3 = symmetry_matrix_3
        symmetry_indices = [0, 1, 3, 3, 2, 0, 3, 2, 4, 1]
        self.symmetry_indices = torch.as_tensor(symmetry_indices, dtype=torch.int64).cuda()
        if self.improve_pose and self.fix_camera:
            self.npcs_unet = SparseUNet.build(
                self.feature_dim, channels[0:3], block_repeat, norm_fn, without_stem=self.feature_dim == channels[0]
            )
            self.npcs_head = nn.Sequential(
                nn.Linear(channels[0], channels[0]),
                norm_fn(channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(channels[0], 3 * (self.num_part_classes - 1)),
            )
        elif self.improve_pose:
            self.npcs_unet = SparseUNet.build(
                self.feature_dim * 2, channels[1:4], block_repeat, norm_fn, without_stem=self.feature_dim * 2 == channels[1]
            )
            self.npcs_head = nn.Sequential(
                nn.Linear(channels[1], channels[0]),
                norm_fn(channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(channels[0], 3 * (self.num_part_classes - 1)),
            )
        else:
            self.npcs_unet = SparseUNet.build(
                self.feature_dim, channels[:2], block_repeat, norm_fn, without_stem=self.feature_dim == channels[0]
            )
            self.npcs_head = nn.Linear(channels[0], 3 * (self.num_part_classes - 1))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        self.device = torch.device('cuda:0')

    def forward(self, pc_pairs, flow_data, do_inference):
        pc1s = [pc_pair.pc1 for pc_pair in pc_pairs] # Point cloud 1 is the primary point cloud
        # if we run gapartnet's method, we just use pc1 for inference
        pc1s = [pc1.to(self.device) for pc1 in pc1s]
        if self.fix_camera:
            pass
        elif self.cat_features:
            # We use the backbone directly to extract features of two point clouds, not flownet. 
            assert flow_data is None, "We use the backbone directly to extract features of two point clouds, not flownet. "
            pc2s = [pc_pair.pc2 for pc_pair in pc_pairs]
            pc2s = [pc2.to(self.device) for pc2 in pc2s]
            for i in range(len(pc2s)):
                pc2s[i] = apply_voxelization(pc2s[i], voxel_size=(1/100, 1/100, 1/100))
        if flow_data is not None:
            flow_data = flow_data.permute(0, 2, 1)
            for i in range(len(pc1s)):
                if not self.cat_points_and_flows:
                    # we directly replace color data with flow data
                    pc1s[i].points[:,3:6] = flow_data[i,:,0:3]
                else:
                    # we cat the color data and flow data [n,6] to [n,9]
                    pc1s[i].points = torch.cat((pc1s[i].points, flow_data[i,:,0:3]), dim=1)
                pc1s[i] = apply_voxelization(pc1s[i], voxel_size=(1/100, 1/100, 1/100))
            # change the color data to flow data if we have, it was inferenced by input contains informations about color
        else:
            assert self.cat_points_and_flows == False, "the in_channels is change to 9, so the flow data is needed"
            for i in range(len(pc1s)):
                pc1s[i] = apply_voxelization(pc1s[i], voxel_size=(1/100, 1/100, 1/100))
        data_batch = PointCloud.collate(pc1s)
        points = data_batch.points
        sem_labels = data_batch.sem_labels
        pc_textual_ids = data_batch.pc_ids
        # for points in an instance: 0-3: mean_xyz; 3-6: max_xyz; 6-9: min_xyz
        instance_regions = data_batch.instance_regions
        instance_labels = data_batch.instance_labels
        batch_indices = data_batch.batch_indices
        # this is compact representation of sem w.r.t. instance.
        instance_sem_labels = data_batch.instance_sem_labels
        num_points_per_instance = data_batch.num_points_per_instance
        gt_npcs = data_batch.gt_npcs
        pt_xyz = points[:, :3]

        # data_batch_copy = copy.deepcopy(data_batch)
        pc_feature = self.forward_backbone(data_batch)
        if not self.offset_cat:
            offsets_preds = self.forward_offset(pc_feature)
            offsets_gt = instance_regions[:, :3] - pt_xyz
        if self.fix_camera:
            # not run flownet for test
            pc_feature_cat = pc_feature
        elif self.cat_features and not self.fix_camera:
            data_batch_2 = PointCloud.collate(pc2s)
            pc_feature_2 = self.forward_backbone(data_batch_2, bone_two=self.two_backbones)
            pt_xyz_2 = data_batch_2.points[:, :3]
            # use flownet like structure to extract features of two point clouds
            # the original input is [bs*n,3](points) and [bs*n,16](features), we change it to [bs,3,n] and [bs,16,n]
            bs = data_batch.batch_size
            pc_feature_flow = self.flownet(pt_xyz.reshape(bs, -1, 3).transpose(1, 2).contiguous(), 
                                              pt_xyz_2.reshape(bs, -1, 3).transpose(1,2).contiguous(), 
                                              pc_feature.reshape(bs, -1, pc_feature.shape[-1]).transpose(1, 2).contiguous(),
                                              pc_feature_2.reshape(bs, -1, pc_feature.shape[-1]).transpose(1, 2).contiguous()).transpose(1,2).reshape(-1, pc_feature.shape[-1]).contiguous()
            pc_feature_cat = torch.cat((pc_feature, pc_feature_flow), dim=1)
            
            # pc_feature = pc_feature_flow
        else:
            pc_feature_cat = pc_feature # just a placeholder
        if self.cat_features:
            sem_logits, sem_seg_features = self.forward_sem_seg(pc_feature_cat, pt_xyz=pt_xyz, batch_size=data_batch.batch_size)
            if self.offset_cat:
                offsets_preds = self.forward_offset(sem_seg_features) # if we cat features, we use sem_seg_features to predict offsets
                offsets_gt = instance_regions[:, :3] - pt_xyz
        else:
            sem_logits = self.forward_sem_seg(pc_feature)
        loss_sem_seg = self.loss_sem_seg(sem_logits, sem_labels)
        sem_preds = sem_logits.detach().argmax(dim=-1)
        loss_offset_dist, loss_offset_dir = self.loss_offset(
            offsets_preds, offsets_gt, sem_labels, instance_labels,  # type: ignore
        )
            
        voxel_tensor, pc_voxel_id, proposals = self.proposal_clustering_and_revoxelize(
            pt_xyz=pt_xyz, batch_indices=batch_indices, pt_features=pc_feature_cat if self.improve_pose else pc_feature,
            sem_preds=sem_preds,
            offset_preds=offsets_preds, 
            instance_labels=instance_labels)

        if proposals is not None:
            proposals.sem_labels = sem_labels[proposals.valid_mask][proposals.sorted_indices]
            proposals.instance_sem_labels = instance_sem_labels
            # note: why not just add voxel feature into proposals?
            pc_features = voxel_tensor.features[pc_voxel_id]
            proposals.pc_features = pc_features

        if (self.current_epoch >= self.start_scorenet or do_inference) and proposals is not None:
            score_logits = self.forward_proposal_score(voxel_tensor, pc_voxel_id, proposals)  # type: ignore
            proposal_offsets_begin = proposals.proposal_offsets[:-1].long()  # type: ignore
            if self.training:
                proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()  # type: ignore
            else:
                proposal_sem_labels = proposals.sem_preds[proposal_offsets_begin].long()  # type: ignore
            # print(proposal_sem_labels.min(), proposal_sem_labels.max())
            score_logits = score_logits.gather(1, proposal_sem_labels[:, None] - 1).squeeze(1)
            proposals.score_preds = score_logits.detach().sigmoid()
            loss_prop_score = self.loss_proposal_score(score_logits, proposals, num_points_per_instance)

            # proposal_offsets_begin = proposals.proposal_offsets[:-1].long()  # type: ignore
            # proposal_sem_labels = proposals.sem_labels[proposal_offsets_begin].long()  # type: ignore
            # print(proposal_sem_labels.min(), proposal_sem_labels.max())
        else:
            loss_prop_score = torch.tensor(0.0).cuda()

        # proposal loss.
        return_dict = {
            'loss_sem_seg': loss_sem_seg,
            'loss_offset_dist': loss_offset_dist,
            'loss_offset_dir': loss_offset_dir,
            'loss_prop_score': loss_prop_score,
        }
        # we manual add something for logs
        return_dict['sem_preds'] = sem_preds.detach() # results detach from calculation
        return_dict['sem_labels'] = sem_labels
        return_dict['proposals'] = proposals

        if (self.current_epoch >= self.start_npcs or do_inference) and proposals is not None:
            npcs_features = self.npcs_unet(voxel_tensor)
            npcs_logits = self.npcs_head(npcs_features.features)
            npcs_logits = npcs_logits[pc_voxel_id]
            gt_npcs = gt_npcs[proposals.valid_mask][proposals.sorted_indices]
            loss_prop_npcs = self.loss_proposal_npcs(npcs_logits, gt_npcs, proposals)
            return_dict['loss_prop_npcs'] = loss_prop_npcs

            if do_inference:
                proposals = filter_invalid_proposals(
                    proposals,
                    score_threshold=self.val_score_threshold,
                    min_num_points_per_proposal=self.val_min_num_points_per_proposal
                )
                proposals = apply_nms(proposals, self.val_nms_iou_threshold)
                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]

                # thetas = self.loss_rotation_thetas(proposals)
                # return_dict['diff_thetas'] = thetas


        return return_dict

    def loss_rotation_thetas(self, proposals):
        gt_npcs = proposals.gt_npcs
        npcs_preds = proposals.npcs_preds
        npcs_valid_mask = proposals.npcs_valid_mask
        proposal_indices = proposals.proposal_indices[npcs_valid_mask]
        pt_xyz = proposals.pt_xyz[npcs_valid_mask]
        sem_classes = proposals.pt_sem_classes

        diff_thetas = []
        # for sem_class, val in zip(proposals.pt_sem_classes, torch.unique(proposal_indices)):
        for val in torch.unique(proposal_indices):
            # sem_class = sem_labels[proposal_indices == val]
            # assert torch.unique(sem_class).shape[0] == 1
            # sem_class = sem_class[0].item()
            sem_class = sem_classes[val].item()
            pt_xyz_ = pt_xyz[proposal_indices == val]
            npcs_preds_ = npcs_preds[proposal_indices == val]
            gt_npcs_ = gt_npcs[proposal_indices == val]
            pred_R = estimate_similarity_transform(npcs_preds_.cpu().numpy(), pt_xyz_.cpu().numpy())[1]
            gt_R = estimate_similarity_transform(gt_npcs_.cpu().numpy(), pt_xyz_.cpu().numpy())[1]

            # import open3d as o3d
            # pcd_npcs = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(npcs_preds_.cpu().numpy()))
            # pcd_npcs.paint_uniform_color([1, 0, 0])
            # pcd_gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt_npcs_.cpu().numpy() + 0.5))
            # pcd_gt.paint_uniform_color([0, 1, 0])
            # o3d.io.write_point_cloud('tmp.ply', pcd_npcs + pcd_gt)

            if pred_R is None or gt_R is None:
                continue

            pred_R, gt_R = pred_R.T, gt_R.T  # change the semantics of R to make diff-theta proper.
            pred_R, gt_R = torch.from_numpy(pred_R).float().cuda(), torch.from_numpy(gt_R).float().cuda()
            symmetry_matrices = get_symmetry_matrices_from_sem_label(sem_class)

            if sem_class == 0:
                assert False
            elif sem_class == 1 or sem_class == 4 or sem_class == 5 or sem_class == 7 or sem_class == 9:
                assert symmetry_matrices.shape[0] == 2
                # sym_gt_R = torch.einsum('sij,jk->sik', symmetry_matrices, gt_R)
                sym_gt_R = torch.einsum('ij,sjk->sik', gt_R, symmetry_matrices)
                rel_R = torch.einsum('sij,jk->sik', sym_gt_R.transpose(-2, -1), pred_R)
                rel_R = rel_R.detach().cpu().numpy()
                traces = np.trace(rel_R, axis1=-2, axis2=-1)
                cos_theta = np.clip(((traces - 1) / 2), -1, 1).max()
                diff_theta = np.arccos(cos_theta) * 180 / np.pi
            elif sem_class == 2 or sem_class == 3 or sem_class == 6:
                assert symmetry_matrices.shape[0] == 12
                z = torch.tensor([0, 0, 1.]).float().cuda()
                z1 = pred_R @ z
                z2 = gt_R @ z
                cos_theta = torch.dot(z1, z2) / (torch.norm(z1) * torch.norm(z2))
                cos_theta = torch.clip(cos_theta, -1, 1)
                diff_theta = torch.acos(cos_theta) * 180 / torch.pi
            elif sem_class == 8:
                z = torch.tensor([0, 0, 1.]).float().cuda()
                z1 = pred_R @ z
                z2 = gt_R @ z
                z2_neg = gt_R @ -z
                cos_theta = torch.max(torch.dot(z1, z2) / (torch.norm(z1) * torch.norm(z2)),
                                      torch.dot(z1, z2_neg) / (torch.norm(z1) * torch.norm(z2_neg)))
                cos_theta = torch.clip(cos_theta, -1, 1)
                diff_theta = torch.acos(cos_theta) * 180 / torch.pi
            else:
                raise NotImplementedError

            diff_theta = diff_theta.item()
            diff_thetas.append(diff_theta)

        return diff_thetas

    def forward_backbone(self, data_batch, bone_two=False):
        voxel_tensor = data_batch.voxel_tensor
        pc_voxel_id = data_batch.pc_voxel_id
        if self.two_backbones and bone_two:
            voxel_features = self.backbone2(voxel_tensor)
        else:
            voxel_features = self.backbone(voxel_tensor)
        pc_feature = voxel_features.features[pc_voxel_id]
        return pc_feature

    def forward_sem_seg(
            self,
            pc_feature: torch.Tensor,  # [20000*bs, feature_dim]
            pt_xyz: torch.Tensor=None,      # [20000*bs, 3]
            batch_size: int=None,            # Batch size
    ) -> torch.Tensor:
        if not self.cat_features:  # just a head is needed.
            sem_logits = self.sem_seg_head(pc_feature)
        else:
            num_points = pt_xyz.shape[0]

            # Calculate batch_offsets
            points_per_batch = num_points // batch_size
            batch_offsets = torch.arange(0, num_points + 1, points_per_batch, dtype=torch.int64, device=pt_xyz.device)

            # Calculate points_range_min and points_range_max
            points_range_min = pt_xyz.min(0)[0] - 1e-4
            points_range_max = pt_xyz.max(0)[0] + 1e-4

            # Perform voxelization
            voxel_features, voxel_coords, batch_indices, pc_voxel_id = voxelize(
                points=pt_xyz,
                pt_features=pc_feature,
                batch_offsets=batch_offsets,
                voxel_size=torch.as_tensor([0.01, 0.01, 0.01], device=pt_xyz.device),
                points_range_min=torch.as_tensor(points_range_min, device=pt_xyz.device),
                points_range_max=torch.as_tensor(points_range_max, device=pt_xyz.device),
                reduction="mean"
            )

            # Ensure all points are voxelized
            assert (pc_voxel_id >= 0).all()

            # Adding batch indices to voxel_coords
            voxel_coords = torch.cat([batch_indices.unsqueeze(1), voxel_coords], dim=1).int() # only support int32

            # Calculate spatial shape
            voxel_coords_range = (voxel_coords[:, 1:].max(0)[0] + 1).clamp(min=128, max=None)
            spatial_shape = voxel_coords_range.tolist()

            # Convert voxel_features and voxel_coords to the format expected by SparseUNet
            # for details: https://github.com/traveller59/spconv/blob/master/docs/USAGE.md
            voxel_features = voxel_features.to(self.device)
            voxel_coords = voxel_coords.to(self.device)

            sparse_input = spconv.SparseConvTensor(
                features=voxel_features,
                indices=voxel_coords,
                spatial_shape=spatial_shape,
                batch_size=batch_size
            )

            # Forward pass through SparseUNet
            sem_seg_features = self.sem_seg_unet(sparse_input)
            sem_seg_features = sem_seg_features.features[pc_voxel_id]
            # Forward pass through sem_seg_head
            sem_logits = self.sem_seg_head(sem_seg_features)
        try:
            return sem_logits, sem_seg_features
        except:
            return sem_logits

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

    def forward_offset(self, pc_feature):
        offset = self.offset_head(pc_feature)
        return offset

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

    def proposal_clustering_and_revoxelize(
            self,
            pt_xyz: torch.Tensor,
            batch_indices: torch.Tensor,
            pt_features: torch.Tensor,
            sem_preds: torch.Tensor,
            offset_preds: torch.Tensor,
            instance_labels: Optional[torch.Tensor],
    ):
        device = torch.device('cuda:0')

        if self.training:
            valid_mask = (sem_preds > 0) & (instance_labels >= 0)
        else:
            valid_mask = (sem_preds > 0)

        # do clustering for valid instances.
        pt_xyz = pt_xyz[valid_mask]
        batch_indices = batch_indices[valid_mask]
        pt_features = pt_features[valid_mask]
        sem_preds = sem_preds[valid_mask].int()
        offset_preds = offset_preds[valid_mask]
        instance_labels = instance_labels[valid_mask]

        _, batch_indices_compact, num_points_per_batch = torch.unique_consecutive(
            batch_indices, return_inverse=True, return_counts=True
        )
        batch_indices_compact = batch_indices_compact.int()
        batch_offsets = torch.zeros(
            (num_points_per_batch.shape[0] + 1,), dtype=torch.int32, device=device
        )
        batch_offsets[1:] = num_points_per_batch.cumsum(0)

        # sorted_cc_labels mean that this is sorted cc labels placed consecutively.
        # sorted_indices means the indices of the points placed.
        sorted_cc_labels, sorted_indices = cluster_proposals(
            pt_xyz, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query,
        )
        sorted_cc_labels_shift, sorted_indices_shift = cluster_proposals(
            pt_xyz + offset_preds, batch_indices_compact, batch_offsets, sem_preds,
            self.ball_query_radius, self.max_num_points_per_query_shift,
        )
        # create two-times more consecutive proposals.
        sorted_cc_labels = torch.cat([
            sorted_cc_labels,
            sorted_cc_labels_shift + sorted_cc_labels.shape[0],
        ], dim=0)
        # sorted-indices point to the indices of points, thus the point indices are shared.
        sorted_indices = torch.cat([sorted_indices, sorted_indices_shift], dim=0)

        # compact the proposal ids.
        # notice that the sorted-cc-labels is the head of the cc, but not the compact cc ids.
        # and thus we can call this proposal indices, good.
        # proposal_indices has the shape (num_points,) but with values (0, ..., num_proposals - 1).
        _, proposal_indices, num_points_per_proposal = torch.unique_consecutive(
            sorted_cc_labels, return_inverse=True, return_counts=True
        )

        # remove small proposals
        valid_proposal_mask = (num_points_per_proposal >= self.min_num_points_per_proposal)
        # valid_proposal_mask is the valid mask for proposals.
        # proposal_indices is the map from proposal to point.
        valid_point_mask = valid_proposal_mask[proposal_indices]

        sorted_indices = sorted_indices[valid_point_mask]
        if sorted_indices.shape[0] == 0:
            return None, None, None

        batch_indices = batch_indices[sorted_indices]
        pt_xyz = pt_xyz[sorted_indices]
        pt_features = pt_features[sorted_indices]
        sem_preds = sem_preds[sorted_indices]
        instance_labels = instance_labels[sorted_indices]

        # re-compact proposal ids.
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

        # get the sparsed version of proposals.
        # voxelization
        voxel_features, voxel_coords, pc_voxel_id, random_offsets = segmented_voxelize(
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

        # # create shape-model and pts association.
        # proposal_offsets_begin = proposal_offsets[:-1]  # type: ignore
        # proposal_offsets_end = proposal_offsets[1:]  # type: ignore
        # proposal_shape_models = []
        # for begin, end in zip(proposal_offsets_begin, proposal_offsets_end):
        #     batch_index = batch_indices[begin: end]
        #     assert batch_index.unique().shape[0] == 1
        #     batch_index = batch_index[0]
        #     inst_label = instance_labels[begin: end]
        #     inst_label, counts = torch.unique(inst_label, return_counts=True)
        #     inst_label = inst_label[counts.argmax()]
        #     shape_model = shape_model_dict[batch_index.item()][inst_label.item()]
        #     proposal_shape_models.append(shape_model)
        # proposal_shape_models = torch.stack(proposal_shape_models, dim=0).float().cuda()
        # proposal_offsets and proposal_indices are basically the same as batch_indices.
        # but these representations are all within the notion of the sparse operation.
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
            # proposal_shape_models=proposal_shape_models,
        )

        return voxel_tensor, pc_voxel_id, proposals

    def loss_proposal_score(
            self,
            score_logits: torch.Tensor,
            proposals: Instances,
            num_points_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        ious = batch_instance_seg_iou(
            proposals.proposal_offsets,  # type: ignore
            proposals.instance_labels,  # type: ignore
            proposals.batch_indices,  # type: ignore
            num_points_per_instance,
        )
        # compute ious w.r.t. all gt instances (num_proposals, num_gt_instances).
        proposals.ious = ious
        proposals.num_points_per_instance = num_points_per_instance

        ious_max = ious.max(-1)[0]
        gt_scores = get_gt_scores(ious_max, 0.75, 0.25)

        return F.binary_cross_entropy_with_logits(score_logits, gt_scores)

    def forward_proposal_score(
            self,
            voxel_tensor: spconv.SparseConvTensor,
            pc_voxel_id: torch.Tensor,
            proposals: Instances,
    ):
        proposal_offsets = proposals.proposal_offsets
        proposal_offsets_begin = proposal_offsets[:-1]  # type: ignore
        proposal_offsets_end = proposal_offsets[1:]  # type: ignore

        # voxel_tensor has a field called indices with (N, 4).
        # the first dimension is batch indices while the last 3 ones are voxel indices.
        score_features = self.score_unet(voxel_tensor)
        score_features = score_features.features[pc_voxel_id]
        pooled_score_features, _ = segmented_maxpool(
            score_features, proposal_offsets_begin, proposal_offsets_end
        )
        score_logits = self.score_head(pooled_score_features)

        return score_logits

    def loss_proposal_npcs(
            self,
            npcs_logits: torch.Tensor,
            gt_npcs: torch.Tensor,
            proposals: Instances,
    ) -> torch.Tensor:
        sem_preds, sem_labels = proposals.sem_preds, proposals.sem_labels
        proposal_indices = proposals.proposal_indices

        if self.training:
            valid_mask = (sem_preds == sem_labels) & (gt_npcs != 0).any(dim=-1)
        else:
            valid_mask = (sem_preds > 0)

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
