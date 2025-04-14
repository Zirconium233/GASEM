import torchmetrics
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from datasets.datasets_pair import PointCloudPair
import numpy as np
from typing import List, Union, Optional
from kornia.metrics import mean_iou as _mean_iou

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

def calculate_pose_metrics(pred_rot_matrices, gt_rot_matrices):
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
def ground_truth_rotations(rot_list: List[torch.Tensor]) -> torch.Tensor:
    rotations = []
    for rot in rot_list:
        # the rotations are stored as 3x3 matrices in pc_pair.rot_1 and pc_pair.rot_2
        rotation_matrix = np.array(rot.cpu())  # Example using rot_1, adjust as needed
        rotations.append(rotation_matrix)
    return torch.tensor(np.stack(rotations))

def points2ballspace(pc_pairs: List[PointCloudPair], inplace=True):
    if not inplace:
        raise NotImplementedError
    for i in range(len(pc_pairs)):
        pc_pairs[i].pc1.points[:,0:3] = (pc_pairs[i].pc1.points[:,0:3] - pc_pairs[i].t1) / pc_pairs[i].s1
        pc_pairs[i].pc2.points[:,0:3] = (pc_pairs[i].pc2.points[:,0:3] - pc_pairs[i].t2) / pc_pairs[i].s2

def rotation_matrix_to_quaternion(R):
    """
    convert a rotation matrix to a quaternion
    params:
    R: (3, 3)
    
    return:
    quaternion (4,)
    """
    q = torch.zeros(4)
    q[0] = torch.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    q[1] = (R[2, 1] - R[1, 2]) / (4.0 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4.0 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4.0 * q[0])
    return q

def quaternion_angle_diff(q1, q2):
    """
    calculate the angle difference between two quaternions
    params:
    q1, q2: (4,)
    
    return:
    the angle difference in degrees
    """
    dot_product = torch.dot(q1, q2)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle_diff_rad = 2 * torch.acos(dot_product)
    angle_diff_deg = torch.rad2deg(angle_diff_rad)
    return angle_diff_deg

def calculate_pose_metrics_quaternion(pred_rot_matrices, gt_rot_matrices):
    """
    calculate the mean rotation error in degrees
    params:
    pred_rot_matrices: (batch_size, 3, 3)
    gt_rot_matrices: (batch_size, 3, 3)
    
    return:
    the mean rotation error in degrees
    """
    batch_size = pred_rot_matrices.size(0)
    rot_errors = []
    for i in range(batch_size):
        pred_quat = rotation_matrix_to_quaternion(pred_rot_matrices[i])
        gt_quat = rotation_matrix_to_quaternion(gt_rot_matrices[i])
        angle_diff = quaternion_angle_diff(pred_quat, gt_quat)
        rot_errors.append(angle_diff)
    mean_rot_error = sum(rot_errors) / batch_size
    return mean_rot_error

@torch.no_grad()
def pixel_accuracy(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Compute pixel accuracy.
    """

    if gt_mask.numel() > 0:
        accuracy = (pred_mask == gt_mask).sum() / gt_mask.numel()
        accuracy = accuracy.item()
    else:
        accuracy = 0.
    return accuracy


@torch.no_grad()
def mean_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, num_classes: int) -> float:
    """
    Compute mIoU.
    """

    valid_mask = gt_mask >= 0
    miou = _mean_iou(
        pred_mask[valid_mask][None], gt_mask[valid_mask][None], num_classes=num_classes
    ).mean()
    return miou


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    if ignore_index is not None:
        valid_mask = targets != ignore_index
        targets = targets[valid_mask]

        if targets.shape[0] == 0:
            return torch.tensor(0.0).to(dtype=inputs.dtype, device=inputs.device)

        inputs = inputs[valid_mask]

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


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

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
