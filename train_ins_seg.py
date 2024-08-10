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
from network.icp import icp, icp_gpu
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
import argparse
# here we import or design our networks
from network.GAPartNetWithFlows import GaPartNetWithFlows
from network.GPVNet import GPVNet
import random

def get_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--root_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/train")
    parser.add_argument('--test_intra_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_intra")
    parser.add_argument('--test_inter_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_inter")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default="log_dir/ins_seg")
    parser.add_argument('--train_with_flow', action='store_true')
    parser.add_argument('--no_train_with_flow', dest='train_with_flow', action='store_false')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no_few_shot', dest='few_shot', action='store_false')
    parser.add_argument('--cat_points_and_flows', action='store_true')
    parser.add_argument('--no_cat_points_and_flows', dest='cat_points_and_flows', action='store_false')
    parser.add_argument('--use_icp', action='store_true')
    parser.add_argument('--random_seed', type=int, default=233)
    parser.add_argument('--icp_original_points', action='store_true')
    parser.add_argument('--icp_transformed_points', dest='icp_original_points', action='store_false')
    parser.add_argument('--gpu_icp', action='store_true')
    parser.add_argument('--cat_features', action='store_true')
    parser.add_argument('--rot_gt', action='store_true')
    parser.add_argument('--cpu_icp', dest='gpu_icp', action='store_false') # We default to use CPU ICP, because the GPU implementation is not stable and take more time than CPU
    parser.add_argument('--gpv_model_path', type=str, default="checkpoints/gpvnet.pth")
    parser.add_argument('--flownet_model_path', type=str, default="checkpoints/flownet.pth")
    parser.add_argument('--improve_pose', action='store_true')
    parser.add_argument('--two_backbones', action='store_true')
    parser.add_argument('--offset_cat', action='store_true')
    parser.set_defaults(train_with_flow=True, 
                        few_shot=False, 
                        cat_points_and_flows=False, 
                        use_icp=False, 
                        icp_original_points=True, 
                        gpu_icp=False, 
                        cat_features=True,
                        rot_gt=False,
                        improve_pose=True,
                        two_backbones=False,
                        offset_cat=False,)
    return parser.parse_args()
    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)

def train_ins_seg(ins_seg, 
                  gpv_net, 
                  flownet, 
                  dataloader_train, 
                  dataloader_test_intra, 
                  dataloader_test_inter, 
                  num_epochs, lr, 
                  log_dir, device, 
                  train_with_flow=True, 
                  icp_original_points=False, 
                  gpu_icp=False,
                  cat_features=False,
                  rot_gt=False,):
    
    # optimizer = torch.optim.Adam(ins_seg.parameters(), lr=lr) # the model contains optimizer
    ins_seg.train()
    gpv_net.eval()
    if flownet is None and not cat_features:
        print("flownet is None, use icp instead")
    elif cat_features:
        print("use backbone to extract features")
    else:
        flownet.eval()
        flownet.to(device)
    ins_seg = ins_seg.to(device)
    gpv_net = gpv_net.to(device)
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
            torch.save(ins_seg.state_dict(), log_dir+r'/'+f"ins_seg_[{epoch+1}|{num_epochs}].pth")
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, device, writer, epoch, 'test_train', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)
        for batch_idx, batch in enumerate(dataloader_train):
            bs = len(batch)
            ins_seg.current_epoch = epoch
            ins_seg.train()
            pc_pairs = [pair.to(device) for pair in batch]
            if train_with_flow:
                with torch.no_grad():
                    if not rot_gt:
                        (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)
                        # Convert predicted vectors and ground truth vectors back to rotation matrices
                        pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
                        pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
                        input1 = torch.stack([(pred_rot_matrices1[i].transpose(0,1) @ pc_pairs[i].pc1.points[:,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
                        feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                        input2 = torch.stack([(pred_rot_matrices2[i].transpose(0,1) @ pc_pairs[i].pc2.points[:,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
                        feat2 = torch.stack([pc_pair.pc2.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    else:
                        # use gt to calculate the input, just for debug
                        R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
                        R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2.T for pc in pc_pairs]))
                        # here we need to collect the rotation matrix
                        gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1, True)
                        gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2, True)
                        input1 = torch.stack([(gt_rot_matrices1[i].cuda().transpose(0,1) @ pc_pairs[i].pc1.points[:,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
                        feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                        input2 = torch.stack([(gt_rot_matrices2[i].cuda().transpose(0,1) @ pc_pairs[i].pc2.points[:,0:3].transpose(0,1)).transpose(0,1) for i in range(bs)], dim=0)
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
            return_dict = ins_seg(pc_pairs, flow_data, do_inference=False)
            loss = sum([return_dict[key] for key in return_dict.keys() if
                'loss' in key and isinstance(return_dict[key], torch.Tensor)])
            ins_seg.optimizer.zero_grad()
            loss.backward()
            ins_seg.optimizer.step()
            # here we need to calculate the precisions
            all_accu = (return_dict['sem_preds'] == return_dict['sem_labels']).sum().float() / (return_dict['sem_labels'].shape[0])
            instance_mask = return_dict['sem_labels'] > 0
            pixel_accu = pixel_accuracy(return_dict['sem_preds'][instance_mask], return_dict['sem_labels'][instance_mask])
            # add step and record precisions
            global_step += 1
            total_loss += loss.item()
            total_all_accu += all_accu.item()
            total_pixel_accu += pixel_accu
            # record loss every 10 batchs
            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/all_accu', all_accu.item(), global_step)
                writer.add_scalar('train/pixel_accu', pixel_accu, global_step)
                try:
                    writer.add_scalar('train/npcs_loss', return_dict['loss_prop_npcs'].item(), global_step)
                except:
                    pass
                print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")
            # torch.cuda.empty_cache()

        avg_loss = total_loss / len(dataloader_train)
        avg_all_accu = total_all_accu / len(dataloader_train)
        avg_pixel_accu = total_pixel_accu / len(dataloader_train)
        print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)
        writer.add_scalar('train/avg_all_accu', avg_all_accu * 100, epoch)
        writer.add_scalar('train/avg_pixel_accu', avg_pixel_accu * 100, epoch)
        if (epoch + 1) % 10 == 0:
            torch.save(ins_seg.state_dict(), log_dir+r'/'+f"ins_seg_[{epoch+1}|{num_epochs}].pth")
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)
            test_ins_seg(ins_seg, gpv_net, flownet, dataloader_train, device, writer, epoch, 'test_train', train_with_flow, icp_original_points=icp_original_points, gpu_icp=gpu_icp, cat_features=cat_features, rot_gt=rot_gt)

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
                 rot_gt=False,):
    
    if rot_gt:
        assert no_rot == False, "if we do not have ground truth rotation, we can not use ground truth rotation to calculate the input"
    print("______________________" + phase + "_______________________")
    ins_seg.eval()
    gpv_net.eval()
    if flownet is None and not cat_features:
        print("flownet is None, use icp instead")
    elif cat_features:
        print("use backbone to extract features")
    else:
        flownet.eval()
    all_sem_preds = []
    all_sem_labels = []
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    all_proposals = []
    mAPs = []
    AP50s = []
    npcs_loss = []
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
                    if not rot_gt:
                        input1 = torch.stack([(pred_rot_matrices1[i].transpose(0,1) @ pc_pairs[i].pc1.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
                        feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                        input2 = torch.stack([(pred_rot_matrices2[i].transpose(0,1) @ pc_pairs[i].pc2.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
                        feat2 = torch.stack([pc_pair.pc2.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    else:
                        # in test method, we still forward the gpv_net to calculate the mean rotation error
                        input1 = torch.stack([(gt_rot_matrices1[i].cuda().transpose(0,1) @ pc_pairs[i].pc1.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
                        feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                        input2 = torch.stack([(gt_rot_matrices2[i].cuda().transpose(0,1) @ pc_pairs[i].pc2.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous() for i in range(bs)], dim=0)
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
                    # remove nan
                    mAP = np.nanmean(mAP) if len(aps) != 0 else 0
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
        print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean AP@50: {(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s]) if not np.isnan(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s])) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mAP: {(np.nanmean(mAPs) if not np.isnan(np.nanmean(mAPs)) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mIoU: {miou * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean All Accu: {mean_all_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean Pixel Accu: {mean_pixel_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean NPCS Loss: {np.mean(npcs_loss):.4f}")
    # record results
    if writer is not None:
        if train_with_flow and not no_rot:
            writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
        writer.add_scalar(
            f"{phase}/mean_AP@50",
            np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s]) * 100 if not np.isnan(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s])) else 0,
            epoch
        )
        writer.add_scalar(
            f"{phase}/mAP",
            np.nanmean(mAPs) * 100 if not np.isnan(np.nanmean(mAPs)) else 0,
            epoch
        )
        writer.add_scalar(
            f"{phase}/mIoU",
            miou * 100,
            epoch
        )
        writer.add_scalar(
            f"{phase}/pixel_accu",
            mean_pixel_accu * 100,
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
        writer.add_scalar(
            f"{phase}/npcs_loss",
            np.mean(npcs_loss),
            epoch
        )
        for class_idx in range(1, 10):
            partname = PART_ID2NAME[class_idx]
            writer.add_scalar(
                f"{phase}/AP@50_{partname}",
                np.nanmean([np.nanmean(ap50[class_idx - 1]) * 100 if ap50 is not None else 0 for ap50 in AP50s]) if not np.isnan(np.nanmean([np.nanmean(ap50[class_idx - 1]) * 100 if ap50 is not None else 0 for ap50 in AP50s])) else 0,
                epoch
            )
    ins_seg.train()

def initialize_weights(model):
    if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight)
        if model.bias is not None:
            nn.init.constant_(model.bias, 0)

def main():
    args = get_args()
    root_dir = args.root_dir
    test_intra_dir = args.test_intra_dir
    test_inter_dir = args.test_inter_dir
    num_epochs = args.num_epochs
    lr = args.lr
    log_dir = args.log_dir
    if args.few_shot:
        log_dir += "_few_shot"
    train_with_flow = args.train_with_flow
    seed = args.random_seed
    gpv_model_path = args.gpv_model_path
    flownet_model_path = args.flownet_model_path
    # set_random_seed(seed)
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=False, shot=args.few_shot, choose_category=None, max_points=20000, augmentation=False)
    dataloader_train, dataloader_test_intra, dataloader_test_inter = get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, num_workers=0, batch_size=8)
    print("len of datasets is: ")
    print(len(dataset_train), len(dataset_test_intra), len(dataset_test_inter))
    print(f"train_with_flow is: {train_with_flow}")
    if train_with_flow:
        if args.cat_features:
            print("cat_features is True")
        else:
            print(f"cat_points_and_flows is: {args.cat_points_and_flows}")
            print(f"use_icp is: {args.use_icp}")
            if args.use_icp:
                print(f"icp_original_points is: {args.icp_original_points}")
                print(f"gpu_icp is: {args.gpu_icp}")
    # init models
    gpv_net = GPVNet().cuda()
    gpv_net.load_state_dict(torch.load(gpv_model_path))
    gpv_net.eval()
    if not args.use_icp and not args.cat_features:
        flownet = FlowNet3D().cuda()
        flownet.load_state_dict(torch.load(flownet_model_path))
        flownet.eval()
    else:
        flownet = None
    if not args.train_with_flow:
        args.improve_pose = False
        args.cat_features = False
    ins_seg = GaPartNetWithFlows(args).cuda()
    ins_seg.train() # only train ins_seg model
    # torch.autograd.set_detect_anomaly(True)
    train_ins_seg(ins_seg, gpv_net, flownet,
                   dataloader_train, dataloader_test_intra, dataloader_test_inter, 
                   num_epochs, lr, log_dir, "cuda:0", train_with_flow, 
                   args.icp_original_points, args.gpu_icp, args.cat_features, args.rot_gt)

if __name__ == "__main__":
    main()