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
from network.icp import icp_mask
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
    parser.add_argument('--root_dir', type=str, default="/16T/zhangran/GAPartNet_fix_small/train")
    parser.add_argument('--test_intra_dir', type=str, default="/16T/zhangran/GAPartNet_fix_small/test_intra")
    parser.add_argument('--test_inter_dir', type=str, default="/16T/zhangran/GAPartNet_fix_small/test_inter")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--log_dir', type=str, default="log_dir/ins_seg_fixed")
    parser.add_argument('--train_with_flow', action='store_true')
    parser.add_argument('--no_train_with_flow', dest='train_with_flow', action='store_false')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no_few_shot', dest='few_shot', action='store_false')
    parser.add_argument('--cat_points_and_flows', action='store_true')
    parser.add_argument('--no_cat_points_and_flows', dest='cat_points_and_flows', action='store_false')
    parser.add_argument('--use_icp', action='store_true')
    parser.add_argument('--random_seed', type=int, default=233)
    parser.add_argument('--cat_features', action='store_true')
    parser.add_argument('--fix_camera', action='store_true')
    parser.add_argument('--gap_and_flow', action='store_false', dest='fix_camera')
    parser.add_argument('--flownet_model_path', type=str, default="checkpoints/flownet.pth")
    parser.set_defaults(train_with_flow=True, 
                        few_shot=False, 
                        cat_points_and_flows=False, 
                        use_icp=True, 
                        cat_features=False,
                        fix_camera=True,
                        )
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
                  flownet, 
                  dataloader_train, 
                  dataloader_test_intra, 
                  dataloader_test_inter, 
                  num_epochs, lr, 
                  log_dir, device, 
                  train_with_flow=True, 
                  cat_features=False,
                  ):
    
    # optimizer = torch.optim.Adam(ins_seg.parameters(), lr=lr) # the model contains optimizer
    ins_seg.train()
    if flownet is None and not cat_features:
        print("flownet is None, use icp instead")
    elif cat_features:
        print("use backbone to extract features")
    else:
        flownet.eval()
        flownet.to(device)
    ins_seg = ins_seg.to(device)
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
            test_ins_seg(ins_seg, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', train_with_flow, cat_features=cat_features)
            test_ins_seg(ins_seg, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', train_with_flow, cat_features=cat_features)
            test_ins_seg(ins_seg, flownet, dataloader_train, device, writer, epoch, 'test_train', train_with_flow, cat_features=cat_features)
        for batch_idx, batch in enumerate(dataloader_train):
            bs = len(batch)
            ins_seg.current_epoch = epoch
            ins_seg.train()
            pc_pairs = [pair.to(device) for pair in batch]
            if train_with_flow:
                with torch.no_grad():
                    # use gt to calculate the input, just for debug
                    # here we need to collect the rotation matrix
                    input1 = torch.stack([(pc_pairs[i].pc1.points[:,0:3] - pc_pairs[i].t1) / pc_pairs[i].s1 for i in range(bs)], dim=0)
                    feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    input2 = torch.stack([(pc_pairs[i].pc2.points[:,0:3] - pc_pairs[i].t1) / pc_pairs[i].s1 for i in range(bs)], dim=0) # use pc2 point but pc1 scale and translation
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
                            for i in range(bs):
                                # just transform the points
                                pc_pairs[i].pc1.points[:,0:3] = input1[i]
                                pc_pairs[i].pc2.points[:,0:3] = input2[i]
                        else:
                            # use icp
                            indices = icp_mask(input1, input2, directly_nn=True) # the pose is same, not need icp, just nearest neighbor
                            batch_indices = torch.arange(indices.size(0)).unsqueeze(-1).expand_as(indices)
                            flow_data = (input2[batch_indices,indices] - input1).permute(0,2,1) # points1 + flow = points2(shuffled)
                            for i in range(bs):
                                # just transform the points
                                pc_pairs[i].pc1.points[:,0:3] = input1[i]
                                pc_pairs[i].pc2.points[:,0:3] = input2[i]
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
            test_ins_seg(ins_seg, flownet, dataloader_test_inter, device, writer, epoch, 'test_inter', train_with_flow, cat_features=cat_features)
            test_ins_seg(ins_seg, flownet, dataloader_test_intra, device, writer, epoch, 'test_intra', train_with_flow, cat_features=cat_features)
            test_ins_seg(ins_seg, flownet, dataloader_train, device, writer, epoch, 'test_train', train_with_flow, cat_features=cat_features)

def test_ins_seg(ins_seg, 
                 flownet, 
                 dataloader, 
                 device, 
                 writer, 
                 epoch, 
                 phase, 
                 train_with_flow=True, 
                 cat_features=False,
                 ):
    
    print("______________________" + phase + "_______________________")
    ins_seg.eval()
    if not train_with_flow:
        print("use color data")
    elif flownet is None and not cat_features:
        print("flownet is None, use icp instead")
    elif cat_features:
        print("use backbone to extract features")
    else:
        flownet.eval()
    all_sem_preds = []
    all_sem_labels = []
    all_proposals = []
    mAPs = []
    AP50s = []
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
                    input1 = torch.stack([(pc_pairs[i].pc1.points[:,0:3] - pc_pairs[i].t1) / pc_pairs[i].s1 for i in range(bs)], dim=0)
                    feat1 = torch.stack([pc_pair.pc1.points[:,3:6] for pc_pair in pc_pairs], dim=0)
                    input2 = torch.stack([(pc_pairs[i].pc2.points[:,0:3] - pc_pairs[i].t1) / pc_pairs[i].s1 for i in range(bs)], dim=0) # use pc2 point but pc1 scale and translation
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
                            for i in range(bs):
                                # just transform the points
                                pc_pairs[i].pc1.points[:,0:3] = input1[i]
                                pc_pairs[i].pc2.points[:,0:3] = input2[i]
                        else:
                            # use icp
                            indices = icp_mask(input1, input2, directly_nn=True) # the pose is same, not need icp, just nearest neighbor
                            batch_indices = torch.arange(indices.size(0)).unsqueeze(-1).expand_as(indices)
                            flow_data = (input2[batch_indices,indices] - input1).permute(0,2,1) # points1 + flow = points2(shuffled)
                            for i in range(bs):
                                # just transform the points
                                pc_pairs[i].pc1.points[:,0:3] = input1[i]
                                pc_pairs[i].pc2.points[:,0:3] = input2[i]
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
    mean_all_accu = total_all_accu / len(dataloader)
    mean_pixel_accu = total_pixel_accu / len(dataloader)
    # print result
    print(f"{phase} - Epoch [{epoch+1}]: Mean AP@50: {(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s]) if not np.isnan(np.nanmean([ap50 if ap50 is not None else 0 for ap50 in AP50s])) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mAP: {(np.nanmean(mAPs) if not np.isnan(np.nanmean(mAPs)) else 0) * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean mIoU: {miou * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean All Accu: {mean_all_accu * 100:.4f}")
    print(f"{phase} - Epoch [{epoch+1}]: Mean Pixel Accu: {mean_pixel_accu * 100:.4f}")
    # record results
    if writer is not None:
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
        for class_idx in range(1, 10):
            partname = PART_ID2NAME[class_idx]
            # use AP50s, AP50s[i] is ap50, but map is not required
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
    flownet_model_path = args.flownet_model_path
    # set_random_seed(seed)
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=False, shot=args.few_shot, choose_category=None, max_points=20000, augmentation=False, no_ball_space=True)
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
    else:
        print("no train with flow")
    if not args.use_icp and not args.cat_features:
        flownet = FlowNet3D().cuda()
        flownet.load_state_dict(torch.load(flownet_model_path))
        flownet.eval()
    else:
        flownet = None

    ins_seg = GaPartNetWithFlows(args).cuda()
    ins_seg.train() # only train ins_seg model
    # torch.autograd.set_detect_anomaly(True)
    train_ins_seg(ins_seg, flownet,
                   dataloader_train, dataloader_test_intra, dataloader_test_inter, 
                   num_epochs, lr, log_dir, "cuda:0", train_with_flow, 
                   args.cat_features)

if __name__ == "__main__":
    main()