import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
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
from datasets.GAPartNet.misc.info import OBJECT_NAME2ID, PART_ID2NAME, PART_NAME2ID, get_symmetry_matrix, SYMMETRY_MATRIX
from datasets.GAPartNet.dataset.instances import Instances
from epic_ops.reduce import segmented_maxpool
from network.gap_grouping_utils import (apply_nms, cluster_proposals, compute_ap,
                               compute_npcs_loss, filter_invalid_proposals,
                               get_gt_scores, segmented_voxelize)
from einops import rearrange, repeat
from epic_ops.iou import batch_instance_seg_iou
from loss.utils import focal_loss, dice_loss, pixel_accuracy, mean_iou
import argparse
from datasets.GAPartNet.misc.pose_fitting import estimate_pose_from_npcs, estimate_similarity_transform
# here we import or design our networks
from network.GAPartNetWithFlows import GaPartNetWithFlows
from network.GPVNet import GPVNet
# from train_ins_seg import test_ins_seg

symmetry_indices = [0, 1, 3, 3, 2, 0, 3, 2, 4, 1]

def get_symmetry_matrices_from_sem_label(sem_label):
    symmetry_index = symmetry_indices[sem_label]
    symmetry_matrices = SYMMETRY_MATRIX[symmetry_index]
    symmetry_matrices = torch.tensor(symmetry_matrices).float().cuda()
    return symmetry_matrices

def estimate_pose_error(proposals):
    gt_npcs = proposals.gt_npcs
    npcs_preds = proposals.npcs_preds
    npcs_valid_mask = proposals.npcs_valid_mask
    proposal_indices = proposals.proposal_indices[npcs_valid_mask]
    pt_xyz = proposals.pt_xyz[npcs_valid_mask]
    sem_classes = proposals.pt_sem_classes

    max_sem_class = 10 # 0 is background
    diff_thetas = [[] for _ in range(max_sem_class)]
    diff_ts = [[] for _ in range(max_sem_class)]
    diff_ss = [[] for _ in range(max_sem_class)]

    for val in torch.unique(proposal_indices):
        sem_class = sem_classes[val].item()
        pt_xyz_ = pt_xyz[proposal_indices == val]
        npcs_preds_ = npcs_preds[proposal_indices == val] - 0.5 # all add a 0.5 offset
        gt_npcs_ = gt_npcs[proposal_indices == val]
        pred_s, pred_R, pred_t, _, _  = estimate_similarity_transform(npcs_preds_.cpu().numpy(), pt_xyz_.cpu().numpy())
        gt_s, gt_R, gt_t, _, _ = estimate_similarity_transform(gt_npcs_.cpu().numpy(), pt_xyz_.cpu().numpy())
        pred_s = pred_s[0]
        gt_s = gt_s[0]
        if pred_R is None or gt_R is None:
            continue

        pred_R, gt_R = pred_R.T, gt_R.T
        pred_R, gt_R = torch.from_numpy(pred_R).float().cuda(), torch.from_numpy(gt_R).float().cuda()
        symmetry_matrices = get_symmetry_matrices_from_sem_label(sem_class)

        if sem_class in {1, 4, 5, 7, 9}:
            sym_gt_R = torch.einsum('ij,sjk->sik', gt_R, symmetry_matrices)
            rel_R = torch.einsum('sij,jk->sik', sym_gt_R.transpose(-2, -1), pred_R)
            rel_R = rel_R.detach().cpu().numpy()
            traces = np.trace(rel_R, axis1=-2, axis2=-1)
            cos_theta = np.clip(((traces - 1) / 2), -1, 1).max()
            diff_theta = np.arccos(cos_theta) * 180 / np.pi
        elif sem_class in {2, 3, 6}:
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
        diff_t = np.linalg.norm(pred_t - gt_t)
        diff_s = abs(pred_s - gt_s)

        diff_thetas[sem_class].append(diff_theta)
        diff_ts[sem_class].append(diff_t)
        diff_ss[sem_class].append(diff_s)

    return diff_thetas, diff_ts, diff_ss

def estimate_pose_error_batch(proposals_list):
    max_sem_class = 10  # 0 is background
    diff_thetas = [[] for _ in range(max_sem_class)]
    diff_ts = [[] for _ in range(max_sem_class)]
    diff_ss = [[] for _ in range(max_sem_class)]

    for proposals in proposals_list:
        thetas, ts, ss = estimate_pose_error(proposals)
        
        for sem_class in range(max_sem_class):
            diff_thetas[sem_class].extend(thetas[sem_class])
            diff_ts[sem_class].extend(ts[sem_class])
            diff_ss[sem_class].extend(ss[sem_class])

    return diff_thetas, diff_ts, diff_ss

def test_ins_seg(ins_seg, 
                 gpv_net, 
                 flownet, 
                 dataloader, 
                 device, 
                 writer, 
                 epoch, 
                 phase, 
                 train_with_flow=True, 
                 no_rot=False, 
                 icp_original_points=False, 
                 gpu_icp=False,
                 cat_features=False,
                 with_pose=False,
                 test_all_ap=False,
                 save_dir=None):
    
    print("______________________" + phase + "_______________________")
    ins_seg.eval()
    gpv_net.eval()
    if flownet is None and not cat_features:
        print("flownet is None, use icp instead")
    elif cat_features:
        print("use backbone to extract features")
    else:
        flownet.eval()
    if test_all_ap:
        save_dir = os.path.join(save_dir, "all_ap")
        os.makedirs(save_dir, exist_ok=True)
        all_ap = [[[] for _ in range(9)] for _ in range(20)]
    all_sem_preds = []
    all_sem_labels = []
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    all_proposals = []
    mAPs = []
    AP50s = []
    npcs_loss = []
    if with_pose:
        diff_thetas = [[] for _ in range(9)]
        diff_ts = [[] for _ in range(9)]
        diff_ss = [[] for _ in range(9)]
    with torch.no_grad():
        total_loss = 0
        total_all_accu = 0
        total_pixel_accu = 0
        batch_idx = 0
        for batch in tqdm(dataloader):
            bs = len(batch)
            ins_seg.current_epoch = epoch
            pc_pairs = [pair.to(device) for pair in batch]
            if train_with_flow:
                with torch.no_grad():
                    (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)
                    pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
                    pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
                    all_pred_rot_matrices.append(pred_rot_matrices1.detach().cpu())
                    all_pred_rot_matrices.append(pred_rot_matrices2.detach().cpu())
                    R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
                    R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2.T for pc in pc_pairs]))
                    # here we need to collect the rotation matrix
                    if not no_rot:
                        gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1, True)
                        gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2, True)
                        all_gt_rot_matrices.append(gt_rot_matrices1.detach().cpu())
                        all_gt_rot_matrices.append(gt_rot_matrices2.detach().cpu())
                    input1 = torch.stack([(pred_rot_matrices1[i].transpose(0,1) @ pc_pairs[i].pc1.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
                    feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    input2 = torch.stack([(pred_rot_matrices2[i].transpose(0,1) @ pc_pairs[i].pc2.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
                    feat2 = torch.stack([pc_pair.pc2.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    if flownet is not None:
                        flow_data = flownet(
                            input1.transpose(1,2).contiguous(),
                            input2.transpose(1,2).contiguous(),
                            feat1.transpose(1,2).contiguous(),
                            feat2.transpose(1,2).contiguous()
                        )
                    else:
                        if cat_features:
                            flow_data = None
                            # rot the points first, and then cat with features
                            for i in range(bs):
                                # use calculated input1 and input2 to avoid double counting
                                pc_pairs[i].pc1.points[:,0:3] = input1[i]
                                pc_pairs[i].pc2.points[:,0:3] = input2[i]
                        else:
                            if not gpu_icp:
                                flow_data = icp(torch.cat([input1, feat1], dim=2), torch.cat([input2, feat2], dim=2), cat_with_color=False, origin_point=icp_original_points).permute(0,2,1)
                            else:
                                flow_data = icp_gpu(torch.cat([input1, feat1], dim=2), torch.cat([input2, feat2], dim=2), cat_with_color=False, origin_point=icp_original_points).permute(0,2,1)
            else:
                flow_data = None
            return_dict = ins_seg(pc_pairs, flow_data, do_inference=True)
            all_accu = (return_dict['sem_preds'] == return_dict['sem_labels']).sum().float() / (return_dict['sem_labels'].shape[0])
            instance_mask = return_dict['sem_labels'] > 0
            pixel_accu = pixel_accuracy(return_dict['sem_preds'][instance_mask], return_dict['sem_labels'][instance_mask])
            loss = sum([return_dict[key] for key in return_dict.keys() if
                        'loss' in key and isinstance(return_dict[key], torch.Tensor)])
            total_loss += loss.item()
            total_all_accu += all_accu.item()
            total_pixel_accu += pixel_accu
            try:
                npcs_loss.append(return_dict['loss_prop_npcs'].item())
            except:
                print("npcs_loss is not calculated")
            batch_idx += 1
            all_sem_preds.append(return_dict['sem_preds'].detach().cpu())
            all_sem_labels.append(return_dict['sem_labels'].detach().cpu())
            if return_dict['proposals'] is not None:
                proposals = return_dict['proposals']
                proposals.pt_sem_classes = proposals.sem_preds[proposals.proposal_offsets[:-1].long()]
                all_proposals.append(proposals)
                if len(all_proposals) == 10 or batch_idx == len(dataloader):
                    # avoid too many proposals taking up too much V-RAM
                    # if just use one proposal, the performance will be bad, many categories may not appear (0 AP)
                    if not test_all_ap:
                        # use 0.5 to 1.0 as the threshold
                        thes = [0.5 + 0.05 * i for i in range(10)]
                    else:
                        # test all of it
                        thes = [0 + 0.05 * i for i in range(20)]
                    aps = []
                    for idx, the in enumerate(thes):
                        if len(all_proposals) != 0:
                            ap = compute_ap(all_proposals, 10, the)
                        else:
                            ap = None
                        if test_all_ap:
                            for sem_class in range(1, 10):
                                if ap is not None:
                                    if not np.isnan(ap[sem_class-1]):
                                        all_ap[idx][sem_class-1].append(ap[sem_class-1]) 
                        if ap is not None:
                            aps.append(ap)
                        if the == 0.5:
                            ap50 = ap
                    mAP = np.array(aps)
                    # remove nan
                    mAP = np.nanmean(mAP) if len(aps) != 0 else 0
                    if with_pose:
                        theta, t, s = estimate_pose_error_batch(all_proposals)
                        for sem_class in range(1, 10):
                            diff_thetas[sem_class-1].extend(theta[sem_class])
                            diff_ts[sem_class-1].extend(t[sem_class])
                            diff_ss[sem_class-1].extend(s[sem_class])
                    all_proposals = []
                    mAPs.append(mAP)
                    AP50s.append(ap50)
            # torch.cuda.empty_cache()
    all_sem_preds = torch.cat(all_sem_preds, dim=0)
    all_sem_labels = torch.cat(all_sem_labels, dim=0)
    miou = mean_iou(all_sem_preds, all_sem_labels, num_classes=10)
    if train_with_flow:
        all_pred_rot_matrices = torch.cat(all_pred_rot_matrices, dim=0)
        if not no_rot:
            all_gt_rot_matrices = torch.cat(all_gt_rot_matrices, dim=0)
            mean_rot_error = calculate_pose_metrics(
                all_pred_rot_matrices, all_gt_rot_matrices
            )
    mean_all_accu = total_all_accu / len(dataloader)
    mean_pixel_accu = total_pixel_accu / len(dataloader)
    # print result
    if train_with_flow and not no_rot:
        print(f"{phase} - Epoch [{epoch+1}]: Mean Base Rotation Error: {mean_rot_error:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean AP@50: {(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s]) if not np.isnan(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s])) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mAP: {(np.nanmean(mAPs) if not np.isnan(np.nanmean(mAPs)) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mIoU: {miou * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean All Accu: {mean_all_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean Pixel Accu: {mean_pixel_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean NPCS Loss: {np.mean(npcs_loss):.4f}")
    # mean_theta = [np.nanmean([np.nanmean(theta[class_idx - 1]) if theta is not None else 0 for theta in theta_errors]) if not np.isnan(np.nanmean([np.nanmean(theta[class_idx - 1]) if theta is not None else 0 for theta in theta_errors])) else 0 for class_idx in range(1, 10)]
    # mean_t = [np.nanmean([np.nanmean(t[class_idx - 1]) if t is not None else 0 for t in t_errors]) if not np.isnan(np.nanmean([np.nanmean(t[class_idx - 1]) if t is not None else 0 for t in t_errors])) else 0 for class_idx in range(1, 10)]
    # mean_s = [np.nanmean([np.nanmean(s[class_idx - 1]) if s is not None else 0 for s in s_errors]) if not np.isnan(np.nanmean([np.nanmean(s[class_idx - 1]) if s is not None else 0 for s in s_errors])) else 0 for class_idx in range(1, 10)]
    if with_pose:
        mean_theta = [np.nanmean(diff_thetas[class_idx - 1]) for class_idx in range(1, 10)]
        mean_t = [np.nanmean(diff_ts[class_idx - 1]) for class_idx in range(1, 10)]
        mean_s = [np.nanmean(diff_ss[class_idx - 1]) for class_idx in range(1, 10)]
        print(f"{phase} - Epoch [{epoch+1}]: Mean Part Rotation Error: {np.nanmean(mean_theta):.4f}")
        print(f"{phase} - Epoch [{epoch+1}]: Mean Part Translation Error: {np.nanmean(mean_t):.4f}")
        print(f"{phase} - Epoch [{epoch+1}]: Mean Part Scale Error: {np.nanmean(mean_s):.4f}")
    # record results
    for class_idx in range(1, 10):
        partname = PART_ID2NAME[class_idx]
        print(f"Class {class_idx} ({partname}):")
        if with_pose:
            print(f"  Mean Rotation Error (Theta): {mean_theta[class_idx-1]:.4f}")
            print(f"  Mean Translation Error (t): {mean_t[class_idx-1]:.4f}")
            print(f"  Mean Scale Error (s): {mean_s[class_idx-1]:.4f}")
        print(f"  AP@50: {np.nanmean([np.nanmean(ap50[class_idx - 1]) * 100 if ap50 is not None else 0 for ap50 in AP50s]) if not np.isnan(np.nanmean([np.nanmean(ap50[class_idx - 1]) * 100 if ap50 is not None else 0 for ap50 in AP50s])) else 0:.4f}")
    if test_all_ap:
        for i in range(20):
            for j in range(9):
                all_ap[i][j] = np.nanmean(all_ap[i][j]) * 100 if not np.isnan(np.nanmean(all_ap[i][j])) else 0
        aps_tensor = torch.tensor(all_ap, dtype=torch.float32)
        pth_path = os.path.join(save_dir, f"ap_{phase}.pth")
        torch.save(aps_tensor, pth_path)
        print(f"All APs are saved: {pth_path}")

def get_args():
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--root_dir', type=str, default="./datasets/GAPartNet/train")
    parser.add_argument('--test_intra_dir', type=str, default="./datasets/GAPartNet/test_intra")
    parser.add_argument('--test_inter_dir', type=str, default="./datasets/GAPartNet/test_inter")
    parser.add_argument('--model_path', type=str, default="log_dir/ins_seg/2024-07-Main-GAP/ins_seg_[300|300].pth")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3) # just a placeholder
    parser.add_argument('--log_dir', type=str, default="log_dir/test")
    parser.add_argument('--use_original_datasets', action='store_true')
    parser.add_argument('--lightning_model', action='store_true')
    parser.add_argument('--train_with_flow', action='store_true')
    parser.add_argument('--no_train_with_flow', dest='train_with_flow', action='store_false')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no_few_shot', dest='few_shot', action='store_false')
    parser.add_argument('--cat_points_and_flows', action='store_true')
    parser.add_argument('--no_cat_points_and_flows', dest='cat_points_and_flows', action='store_false')
    parser.add_argument('--use_origin_points', action='store_true')
    parser.add_argument('--with_pose', action='store_true')
    parser.add_argument('--cat_features', action='store_true')
    parser.add_argument('--improve_pose', action='store_true')
    parser.add_argument('--offset_cat', action='store_true')
    parser.add_argument('--test_all_ap', action='store_true')
    parser.add_argument('--gpv_model_path', type=str, default="checkpoints/gpvnet.pth")
    parser.add_argument('--flownet_model_path', type=str, default="checkpoints/flownet.pth")
    parser.set_defaults(train_with_flow=True, 
                        few_shot=False, 
                        cat_points_and_flows=False, 
                        use_original_datasets=False, 
                        lightning_model=False, 
                        use_icp=False, 
                        use_origin_points=True, 
                        use_origin_flows=False,
                        with_pose=False,
                        cat_features=False,
                        improve_pose=False,
                        offset_cat=False,
                        test_all_ap=False)
    return parser.parse_args()

def test_ins_seg_all(ins_seg, gpv_net, flownet, dataloader_test_intra, dataloader_test_inter, train_with_flow, use_origin_points=False, cat_features=False, with_pose=True, test_all_ap=True, save_dir="log_dir/test"):
    name = str(datetime.today())
    test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, epoch=100, phase="test_intra", train_with_flow=train_with_flow, writer=None, device="cuda:0", icp_original_points=use_origin_points, cat_features=cat_features, with_pose=with_pose, test_all_ap=test_all_ap, save_dir=os.path.join(save_dir, name))
    test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, epoch=100, phase="test_inter", train_with_flow=train_with_flow, writer=None, device="cuda:0", icp_original_points=use_origin_points, cat_features=cat_features, with_pose=with_pose, test_all_ap=test_all_ap, save_dir=os.path.join(save_dir, name))

def main():
    args = get_args()
    root_dir = args.root_dir
    test_intra_dir = args.test_intra_dir
    test_inter_dir = args.test_inter_dir
    train_with_flow = args.train_with_flow
    voxelization = False if train_with_flow else True
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=voxelization, shot=args.few_shot, choose_category=None, max_points=20000, augmentation=False, with_pose=not args.use_original_datasets)
    dataloader_train, dataloader_test_intra, dataloader_test_inter = get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, num_workers=0, batch_size=8)
    print("len of datasets is: ")
    print(len(dataset_train), len(dataset_test_intra), len(dataset_test_inter))
    print(f"with pose is {not args.use_original_datasets}")
    print(f"train_with_flow is: {train_with_flow}")
    print(f"cat_points_and_flows is: {args.cat_points_and_flows}")
    log_dir = args.log_dir
    if args.few_shot:
        log_dir = log_dir + "_few_shot"
    gpv_net = GPVNet().cuda()
    gpv_net.load_state_dict(torch.load(args.gpv_model_path))
    gpv_net.eval()

    if not args.use_icp and not args.cat_features:
        flownet = FlowNet3D().cuda()
        flownet.load_state_dict(torch.load(args.flownet_model_path))
        flownet.eval()
    else:
        flownet = None

    ins_seg = GaPartNetWithFlows(args).cuda()
    if not args.lightning_model:
        ins_seg.load_state_dict(torch.load(args.model_path))
    else:
        ins_seg.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)
    ins_seg.eval()
    if args.with_pose:
        ins_seg.min_num_points_per_proposal = 50 # 3 default
        ins_seg.val_score_threshold = 0.3 # 0.09 default
        ins_seg.val_nms_iou_threshold = 0.5 # 0.3 default
    test_ins_seg_all(ins_seg, gpv_net, flownet, dataloader_test_intra, dataloader_test_inter, train_with_flow, args.use_origin_points, args.cat_features, args.with_pose, args.test_all_ap, save_dir=log_dir)

if __name__ == "__main__":
    main()