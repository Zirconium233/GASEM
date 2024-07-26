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
from train_ins_seg import test_ins_seg

def get_args():
    parser = argparse.ArgumentParser(description='Testing script')
    parser.add_argument('--root_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/train")
    parser.add_argument('--test_intra_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_intra")
    parser.add_argument('--test_inter_dir', type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_inter")
    parser.add_argument('--model_path', type=str, default="log_dir/ins_seg/2024-07-12 22:19:26.945216/ins_seg_[100|100].pth")
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3) # just a placeholder
    parser.add_argument('--log_dir', type=str, default="log_dir/ins_seg")
    parser.add_argument('--test_pose', action='store_true')
    parser.add_argument('--use_original_datasets', action='store_true')
    parser.add_argument('--lightning_model', action='store_true')
    parser.add_argument('--train_with_flow', action='store_true')
    parser.add_argument('--no_train_with_flow', dest='train_with_flow', action='store_false')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no_few_shot', dest='few_shot', action='store_false')
    parser.add_argument('--cat_points_and_flows', action='store_true')
    parser.add_argument('--no_cat_points_and_flows', dest='cat_points_and_flows', action='store_false')
    parser.add_argument('--use_origin_points', action='store_true')
    parser.add_argument('--use_gt', action='store_true')
    parser.add_argument('--cat_features', action='store_true')
    parser.add_argument('--gpv_model_path', type=str, default="checkpoints/gpvnet.pth")
    parser.add_argument('--flownet_model_path', type=str, default="checkpoints/flownet.pth")
    parser.set_defaults(train_with_flow=True, 
                        few_shot=False, 
                        cat_points_and_flows=False, 
                        test_pose=False, 
                        use_original_datasets=False, 
                        lightning_model=False, 
                        use_icp=True, 
                        use_origin_points=True, 
                        use_origin_flows=False,
                        use_gt=False,
                        cat_features=False)
    return parser.parse_args()

def test_ins_seg_all(ins_seg, gpv_net, flownet, dataloader_test_intra, dataloader_test_inter, train_with_flow, use_origin_points=False, cat_features=False, use_gt=False):
    test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_intra, epoch=100, phase="test_intra", train_with_flow=train_with_flow, writer=None, device="cuda:0", icp_original_points=use_origin_points, cat_features=cat_features, use_gt=use_gt)
    test_ins_seg(ins_seg, gpv_net, flownet, dataloader_test_inter, epoch=100, phase="test_inter", train_with_flow=train_with_flow, writer=None, device="cuda:0", icp_original_points=use_origin_points, cat_features=cat_features, use_gt=use_gt)

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

    gpv_net = GPVNet().cuda()
    gpv_net.load_state_dict(torch.load(args.gpv_model_path))
    gpv_net.eval()

    if not args.use_icp:
        flownet = FlowNet3D().cuda()
        flownet.load_state_dict(torch.load(args.flownet_model_path))
        flownet.eval()
    else:
        flownet = None
    flownet.eval()

    ins_seg = GaPartNetWithFlows(args).cuda()
    if not args.lightning_model:
        ins_seg.load_state_dict(torch.load(args.model_path))
    else:
        ins_seg.load_state_dict(torch.load(args.model_path)['state_dict'], strict=False)
    ins_seg.eval()
    # debug
    # ins_seg.min_num_points_per_proposal = 16
    if not args.test_pose:
        test_ins_seg_all(ins_seg, gpv_net, flownet, dataloader_test_intra, dataloader_test_inter, train_with_flow, args.use_origin_points, args.cat_features, args.use_gt)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()