import visu.visu as visu
import torch
import torch.nn as nn
from typing import List, Dict, Union
from torch.utils.data import Dataset
from network.utils import vectors_to_rotation_matrix, get_gt_v
from loss.utils import calculate_pose_metrics, calculate_pose_metrics_quaternion
import random
import cv2
import numpy as np
from datasets.GAPartNet.misc.visu_util import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 
import matplotlib.pyplot as plt
from network.icp import icp, icp_gpu, icp_mask
# only gpvpose is available


def random_show(model: nn.Module, datasets: List[Dataset], dir_name: List[str], log_name: Dict, i: int = None):
    for dataset in datasets:
        model.eval()
        i = random.randint(0, len(dataset)) if i == None or len(datasets) != 1 else i
        print(i)
        inputs = dataset[i]

        name_1 = dataset.group_files[i][0].split('/')[-1].split('.')[0]
        name_2 = dataset.group_files[i][1].split('/')[-1].split('.')[0]
        with torch.no_grad():
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model([inputs])
        rot_1_pred = vectors_to_rotation_matrix(p_green_R1, p_red_R1)
        rot_2_pred = vectors_to_rotation_matrix(p_green_R2, p_red_R2)
        visu.visualize_gapartnet(f"./log_dir/GPV_test/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_1, rot_pred = rot_1_pred.detach().squeeze(0).cpu(), five=False)
        visu.visualize_gapartnet(f"./log_dir/GPV_test/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_2, rot_pred = rot_2_pred.detach().squeeze(0).cpu(), five=False)
    
def random_check_pair(model: nn.Module, datasets: List[Dataset], dir_name: List[str], log_name: Dict, i: int = None, transpose = False):
    for dataset in datasets:
        model.eval()
        i = random.randint(0, len(dataset)) if i == None or len(datasets) != 1 else i
        print('index: ', i)
        print("datasets: ", dir_name[dataset])
        inputs = dataset[i]
        name_1 = dataset.group_files[i][0].split('/')[-1].split('.')[0]
        name_2 = dataset.group_files[i][1].split('/')[-1].split('.')[0]
        print(name_1, ' and ', name_2)
        with torch.no_grad():
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model([inputs])
        rot_1_pred = vectors_to_rotation_matrix(p_green_R1, p_red_R1, transpose)
        rot_1_gt = inputs.rot_1.unsqueeze(0) if not transpose else inputs.rot_1.unsqueeze(0).transpose(1, 2)
        print("rot1: ", rot_1_pred)
        print("gt1: ", rot_1_gt)
        print("vectors1: ", p_green_R1, p_red_R1)
        R_green_gt1, R_red_gt1 = get_gt_v(rot_1_gt)
        print("vectors_gt1: ", R_green_gt1, R_red_gt1)
        error1 = calculate_pose_metrics(rot_1_pred, rot_1_gt)
        error_gt = calculate_pose_metrics(rot_1_gt, rot_1_gt)
        print("error1: ", error1)
        print("error_gt: ", error_gt)
        rot_2_pred = vectors_to_rotation_matrix(p_green_R2, p_red_R2, transpose)
        rot_2_gt = inputs.rot_2.unsqueeze(0) if not transpose else inputs.rot_2.unsqueeze(0).transpose(1, 2)
        print("rot2: ", rot_2_pred)
        print("gt2: ", rot_2_gt)
        print("vectors2: ", p_green_R2, p_red_R2)
        R_green_gt2, R_red_gt2 = get_gt_v(rot_2_gt)
        print("vectors_gt2: ", R_green_gt2, R_red_gt2)
        error2 = calculate_pose_metrics(rot_2_pred, rot_2_gt)
        error_gt_2 = calculate_pose_metrics(rot_2_gt, rot_2_gt)
        print("error2: ", error2)
        print("error_gt: ", error_gt_2)
        visu.visualize_gapartnet(f"./log_dir/GPV_test_sym_v1/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_1, rot_pred = rot_1_pred.detach().squeeze(0).cpu(), five=False)
        visu.visualize_gapartnet(f"./log_dir/GPV_test_sym_v1/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_2, rot_pred = rot_2_pred.detach().squeeze(0).cpu(), five=False)
        # show_merged_pair(dir_name[dataset], name_1, name_2, rot_1_pred, rot_2_pred)

def show_merged_pair(    
    GAPARTNET_DATA_ROOT,
    name_1: str = "pc",
    name_2: str = "pc",
    # t1: Union[torch.Tensor, np.ndarray] = None,
    # t2: Union[torch.Tensor, np.ndarray] = None,
    # use_tran: bool = True,
    merge_with_t: bool = False,
    use_origin_color = True
):
    
    points_input1, meta_all1, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

    points_input1 = points_input1.numpy()
    points_input2= points_input2.numpy()
    xyz_input1 = points_input1[:,:3]
    rgb1 = points_input1[:,3:6]
    trans1 = meta_all1['scale_param']
    xyz_input2 = points_input2[:,:3]
    rgb2 = points_input2[:,3:6]
    trans2 = meta_all2['scale_param']
    t1 = meta_all1['camera2world_translation']
    t2 = meta_all2['camera2world_translation']
    xyz1 = xyz_input1 * trans1[0] + trans1[1:4]
    xyz2 = xyz_input2 * trans2[0] + trans2[1:4]

    if not use_origin_color:
        # use red for image 1 and blue for image 2
        rgb1[:,0:3] = np.array([1, 0, 0])  # red
        rgb2[:,0:3] = np.array([0, 0, 1])  # blue

    if merge_with_t:
        xyz1 = xyz1
        xyz2 = xyz2 - t2 + t1

    # pc_img1 = map2image(xyz1, rgb1*255.0)
    # pc_img1 = cv2.cvtColor(pc_img1, cv2.COLOR_BGR2RGB)
    # pc_img2 = map2image(xyz2, rgb2*255.0)
    # pc_img2 = cv2.cvtColor(pc_img2, cv2.COLOR_BGR2RGB)

    xyz_world_ball1 = visu.rotation_to_world(xyz_input1, np.array(meta_all1["world2camera_rotation"]).reshape(3, 3))
    xyz_world_ball2 = visu.rotation_to_world(xyz_input2, np.array(meta_all2["world2camera_rotation"]).reshape(3, 3))
    if merge_with_t:
        xyz_world_ball2 = xyz_world_ball2 * trans2[0] + trans2[1:4]
        xyz_world_ball2 = xyz_world_ball2 - t2 + t1 - trans2[1:4]
        xyz_world_ball2 = xyz_world_ball2 / trans2[0]
    xyz_world_ball = np.concatenate((xyz_world_ball1, xyz_world_ball2), axis=0)
    rgb = np.concatenate((rgb1, rgb2), axis=0)
    xyz_world = xyz_world_ball * trans1[0] + trans1[1:4]
    pc_img_world = map2image(xyz_world, rgb*255.0)
    pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
    plt.imshow(pc_img_world)
    # cv2.imwrite(f"{final_save_root}/{name}.png", final_img)

def show_merged_pair_with_pred_rotation(
    GAPARTNET_DATA_ROOT,
    rot1: np.ndarray,
    rot2: np.ndarray,
    name_1: str = "pc",
    name_2: str = "pc",
    merge_with_t: bool = False,
    use_origin_color = True
):
    points_input1, meta_all1, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

    points_input1 = points_input1.numpy()
    points_input2 = points_input2.numpy()
    xyz_input1 = points_input1[:,:3]
    rgb1 = points_input1[:,3:6]
    trans1 = meta_all1['scale_param']
    xyz_input2 = points_input2[:,:3]
    rgb2 = points_input2[:,3:6]
    trans2 = meta_all2['scale_param']
    t1 = meta_all1['camera2world_translation']
    t2 = meta_all2['camera2world_translation']

    # rot by given rotation
    xyz_world_ball1 = visu.rotation_to_world(xyz_input1, rot1)
    xyz_world_ball2 = visu.rotation_to_world(xyz_input2, rot2)

    if not use_origin_color:
        rgb1[:,0:3] = np.array([1, 0, 0])  # red
        rgb2[:,0:3] = np.array([0, 0, 1])  # blue

    if merge_with_t:
        xyz_world_ball2 = xyz_world_ball2 - t2 + t1

    xyz_world_ball = np.concatenate((xyz_world_ball1, xyz_world_ball2), axis=0)
    rgb = np.concatenate((rgb1, rgb2), axis=0)
    xyz_world = xyz_world_ball * trans1[0] + trans1[1:4]
    pc_img_world = map2image(xyz_world, rgb*255.0)
    pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
    plt.imshow(pc_img_world)

def show_datasets_pair(GAPARTNET_DATA_ROOT, dataset: Dataset, i: int = None, roted = False):
    if i is None:
        i = random.randint(0, len(dataset))
    print('index: ', i)
    inputs = dataset[i]
    name_1 = dataset.group_files[i][0].split('/')[-1].split('.')[0]
    name_2 = dataset.group_files[i][1].split('/')[-1].split('.')[0]
    print("names: ", name_1, " and ", name_2)
    print("rot1: ", inputs.rot_1)
    print("rot2: ", inputs.rot_2)
    points_input1, meta_all1, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)
    
    points_input1_ = points_input1.numpy()
    points_input2_ = points_input2.numpy()

    p1 = inputs.pc1.points.numpy()
    p2 = inputs.pc2.points.numpy()

    
    xyz_input1 = p1[:,:3]
    rgb1 = p1[:,3:6]
    trans1 = meta_all1['scale_param']
    xyz_input2 = p2[:,:3]
    rgb2 = p2[:,3:6]
    trans2 = meta_all2['scale_param']

    if roted:
        xyz_input1 = visu.rotation_to_world(xyz_input1, inputs.rot_1)
        xyz_input2 = visu.rotation_to_world(xyz_input2, inputs.rot_2)

    xyz1 = xyz_input1 * trans1[0] + trans1[1:4]
    xyz2 = xyz_input2 * trans2[0] + trans2[1:4]
    pc_img1 = map2image(xyz1, rgb1*255.0)
    pc_img1 = cv2.cvtColor(pc_img1, cv2.COLOR_BGR2RGB)
    pc_img2 = map2image(xyz2, rgb2*255.0)
    pc_img2 = cv2.cvtColor(pc_img2, cv2.COLOR_BGR2RGB)

    xyz_input1_ = points_input1_[:,:3]
    rgb1_ = points_input1_[:,3:6]
    xyz_input2_ = points_input2_[:,:3]
    rgb2_ = points_input2_[:,3:6]

    if roted:
        xyz_input1_ = visu.rotation_to_world(xyz_input1_, torch.tensor(meta_all1["world2camera_rotation"]).reshape(3, 3))
        xyz_input2_ = visu.rotation_to_world(xyz_input2_, torch.tensor(meta_all2["world2camera_rotation"]).reshape(3, 3))

    xyz1_ = xyz_input1_ * trans1[0] + trans1[1:4]
    xyz2_ = xyz_input2_ * trans2[0] + trans2[1:4]
    pc_img1_ = map2image(xyz1_, rgb1_*255.0)
    pc_img1_ = cv2.cvtColor(pc_img1_, cv2.COLOR_BGR2RGB)
    pc_img2_ = map2image(xyz2_, rgb2_*255.0)
    pc_img2_ = cv2.cvtColor(pc_img2_, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(pc_img1)
    plt.subplot(2,2,2)
    plt.imshow(pc_img2)
    plt.subplot(2,2,3)
    plt.imshow(pc_img1_)
    plt.subplot(2,2,4)
    plt.imshow(pc_img2_)
    plt.show()

def visu_flows(flow_field: torch.Tensor):
    """
    Visualize flow field in 3D
    params:
    - flow_field: shape: (bs,3,n) : torch.Tensor
    """
    bs = flow_field.size(0)
    assert flow_field.size(1) == 3, "Flow field must have 3 channels"
    n = flow_field.size(2)
    for i in range(bs):
        x = flow_field[i, 0, :].cpu().detach().numpy()
        y = flow_field[i, 1, :].cpu().detach().numpy()
        z = flow_field[i, 2, :].cpu().detach().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x, y, z, x, y, z, length=0.1)
        plt.show()
        if i != 0:
            break # 只显示第一个Batch的结果

def visu_flows_with_start_points(flow_field: torch.Tensor, start_points: torch.Tensor):
    """
    Visualize flow field in 3D
    params:
    - flow_field: shape: (bs,3,n) : torch.Tensor
    - start_points: shape: (bs,3,n) : torch.Tensor
    """
    bs = flow_field.size(0)
    assert flow_field.size(1) == 3, "Flow field must have 3 channels"
    n = flow_field.size(2)
    for i in range(bs):
        x = flow_field[i, 0, :].cpu().detach().numpy()
        y = flow_field[i, 1, :].cpu().detach().numpy()
        z = flow_field[i, 2, :].cpu().detach().numpy()
        x_start = start_points[i, 0, :].cpu().detach().numpy()
        y_start = start_points[i, 1, :].cpu().detach().numpy()
        z_start = start_points[i, 2, :].cpu().detach().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(x_start, y_start, z_start, x, y, z, length=0.1)
        plt.show()
        if i != 0:
            break # 只显示第一个Batch的结果

def random_show_flow(gpv_net: nn.Module, flownet: nn.Module, dataset: Dataset, dir_name: Dict, use_gt=False, use_icp=False, idx=None, icp_device="gpu"):
    if gpv_net is not None:
        gpv_net.cuda()
    else:
        print("gpv_net is None, no rotation prediction")
    if flownet is not None:
        flownet.cuda()
    else:
        assert use_icp, "flownet is None, no flow prediction"
    if idx is None:
        i = random.randint(0, len(dataset))
    else:
        i = idx
    pc_pairs = [dataset[i].to("cuda:0")]
    with torch.no_grad():
        if gpv_net is not None:
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)
            rot_1_pred = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
            rot_2_pred = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
            error1 = calculate_pose_metrics(rot_1_pred, torch.stack([pc_pairs[0].rot_1.transpose(0,1)]))
            print("error1: ", error1)
            error2 = calculate_pose_metrics(rot_2_pred, torch.stack([pc_pairs[0].rot_2.transpose(0,1)]))
            print("error2: ", error2)
            input1 = (rot_1_pred[0].transpose(0,1) @ pc_pairs[0].pc1.points[0:2048,0:3].transpose(0,1)).transpose(0,1)
            feat1 = pc_pairs[0].pc1.points[0:2048,3:6]
            input2 = (rot_2_pred[0].transpose(0,1) @ pc_pairs[0].pc2.points[0:2048,0:3].transpose(0,1)).transpose(0,1)
            feat2 = pc_pairs[0].pc2.points[0:2048,3:6]
        else:
            # directly not use rotation
            if not use_gt:
                rot_1_pred = torch.eye(3).unsqueeze(0).to("cuda:0")
                rot_2_pred = torch.eye(3).unsqueeze(0).to("cuda:0")
                input1 = pc_pairs[0].pc1.points[0:2048,0:3]
                feat1 = pc_pairs[0].pc1.points[0:2048,3:6]
                input2 = pc_pairs[0].pc2.points[0:2048,0:3]
                feat2 = pc_pairs[0].pc2.points[0:2048,3:6]
            else:
                # use gt rotation
                rot_1_pred = pc_pairs[0].rot_1.unsqueeze(0).to("cuda:0")
                rot_2_pred = pc_pairs[0].rot_2.unsqueeze(0).to("cuda:0")
                input1 = (rot_1_pred[0].transpose(0,1) @ pc_pairs[0].pc1.points[0:2048,0:3].transpose(0,1)).transpose(0,1)
                feat1 = pc_pairs[0].pc1.points[0:2048,3:6]
                input2 = (rot_2_pred[0].transpose(0,1) @ pc_pairs[0].pc2.points[0:2048,0:3].transpose(0,1)).transpose(0,1)
                feat2 = pc_pairs[0].pc2.points[0:2048,3:6]
        if not use_icp:
            output = flownet(
                input1.unsqueeze(0).transpose(1,2).contiguous(),
                input2.unsqueeze(0).transpose(1,2).contiguous(),
                feat1.unsqueeze(0).transpose(1,2).contiguous(),
                feat2.unsqueeze(0).transpose(1,2).contiguous()
            )
        else:
            if icp_device == "gpu":
                output = icp_gpu(torch.cat([input1, feat1], dim=1).unsqueeze(0), torch.cat([input2, feat2], dim=1).unsqueeze(0))
            else:
                output = icp(torch.cat([input1, feat1], dim=1).unsqueeze(0), torch.cat([input2, feat2], dim=1).unsqueeze(0))
    if not use_icp:
        visu_flows(output)
    else:
        visu_flows(output.permute(0,2,1))
    if not use_gt:
        show_merged_pair_with_pred_rotation(dir_name[dataset], 
                                            rot_1_pred[0].transpose(0,1).detach().cpu().numpy(), 
                                            rot_2_pred[0].transpose(0,1).detach().cpu().numpy(), 
                                            dataset.group_files[i][0].split('/')[-1].split('.')[0], 
                                            dataset.group_files[i][1].split('/')[-1].split('.')[0], use_origin_color=False)
    else:
        show_merged_pair_with_pred_rotation(dir_name[dataset], 
                                            pc_pairs[0].rot_1.cpu().numpy(), 
                                            pc_pairs[0].rot_2.cpu().numpy(), 
                                            dataset.group_files[i][0].split('/')[-1].split('.')[0], 
                                            dataset.group_files[i][1].split('/')[-1].split('.')[0], use_origin_color=False)

def view_pc1_to_pc2(
    GAPARTNET_DATA_ROOT,
    rot1: np.ndarray,
    rot2: np.ndarray,
    name_1: str = "pc",
    name_2: str = "pc",
    merge_with_t: bool = False,
    flow_data: np.ndarray = np.array([0, 0, 0]),
    use_origin_color = True
):
    points_input1, meta_all1, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = visu.process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

    points_input1 = points_input1.numpy()
    points_input2 = points_input2.numpy()
    xyz_input1 = points_input1[:,:3]
    rgb1 = points_input1[:,3:6]
    trans1 = meta_all1['scale_param']
    xyz_input2 = points_input2[:,:3]
    rgb2 = points_input2[:,3:6]
    trans2 = meta_all2['scale_param']
    t1 = meta_all1['camera2world_translation']
    t2 = meta_all2['camera2world_translation']

    xyz_world_ball1 = visu.rotation_to_world(xyz_input1, rot1)
    xyz_world_ball2 = visu.rotation_to_world(xyz_input2, rot2)

    if not use_origin_color:
        rgb1[:,0:3] = np.array([1, 0, 0])  # red
        rgb2[:,0:3] = np.array([0, 0, 1])  # blue

    if merge_with_t:
        xyz_world_ball2 = xyz_world_ball2 - t2 + t1

    xyz_world_ball = np.concatenate((xyz_world_ball1 + flow_data, xyz_world_ball2), axis=0)
    rgb = np.concatenate((rgb1, rgb2), axis=0)
    xyz_world = xyz_world_ball * trans1[0] + trans1[1:4]
    pc_img_world = map2image(xyz_world, rgb*255.0)
    pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
    plt.imshow(pc_img_world)

def random_show_flow_pc1_to_pc2(gpv_net: nn.Module, flownet: nn.Module, dataset: Dataset, dir_name: Dict, use_gt=False, use_icp=False, idx=None, icp_device="gpu"):
    gpv_net.cuda()
    if idx is None:
        i = random.randint(0, len(dataset))
    else:
        i = idx
    pc_pairs = [dataset[i].to("cuda:0")]
    with torch.no_grad():
        (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net(pc_pairs)
        rot_1_pred = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
        rot_2_pred = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
        error1 = calculate_pose_metrics(rot_1_pred, torch.stack([pc_pairs[0].rot_1.transpose(0,1)]))
        print("error1: ", error1)
        error2 = calculate_pose_metrics(rot_2_pred, torch.stack([pc_pairs[0].rot_2.transpose(0,1)]))
        print("error2: ", error2)
        input1 = (rot_1_pred[0].transpose(0,1) @ pc_pairs[0].pc1.points[:,0:3].transpose(0,1)).transpose(0,1)
        feat1 = pc_pairs[0].pc1.points[:,3:6]
        input2 = (rot_2_pred[0].transpose(0,1) @ pc_pairs[0].pc2.points[:,0:3].transpose(0,1)).transpose(0,1)
        feat2 = pc_pairs[0].pc2.points[:,3:6]
        if not use_icp:
            output = flownet(
                input1.unsqueeze(0).transpose(1,2).contiguous(),
                input2.unsqueeze(0).transpose(1,2).contiguous(),
                feat1.unsqueeze(0).transpose(1,2).contiguous(),
                feat2.unsqueeze(0).transpose(1,2).contiguous()
            )
        else:
            if icp_device == "gpu":
                output = icp_gpu(torch.cat([input1, feat1], dim=1).unsqueeze(0), torch.cat([input2, feat2], dim=1).unsqueeze(0))
            else:
                output = icp(torch.cat([input1, feat1], dim=1).unsqueeze(0), torch.cat([input2, feat2], dim=1).unsqueeze(0))
    if use_icp:
        output = output.permute(0,2,1).cpu()
    visu_flows(output[:,0:3,0:2048].cpu())
    if not use_gt:
        view_pc1_to_pc2(dir_name[dataset], 
                        rot_1_pred[0].transpose(0,1).detach().cpu().numpy(), 
                        rot_2_pred[0].transpose(0,1).detach().cpu().numpy(), 
                        dataset.group_files[i][0].split('/')[-1].split('.')[0], 
                        dataset.group_files[i][1].split('/')[-1].split('.')[0],
                        flow_data=output[0,0:3,:].permute(1,0).cpu().numpy(), 
                        use_origin_color=False)
    else:
        view_pc1_to_pc2(dir_name[dataset], 
                        pc_pairs[0].rot_1.cpu().numpy(), 
                        pc_pairs[0].rot_2.cpu().numpy(), 
                        dataset.group_files[i][0].split('/')[-1].split('.')[0], 
                        dataset.group_files[i][1].split('/')[-1].split('.')[0],
                        flow_data=output[0,0:3,:].permute(1,0).cpu().numpy(), 
                        use_origin_color=False)


def random_show_fixed_camera(points_input1, points_input2, flownet=None, use_origin_color=False, add_flow=False, directly_nn=False, trans: np.ndarray = None):
    points_input1 = points_input1.numpy()
    points_input2 = points_input2.numpy()
    xyz_input1 = points_input1[:,:3]
    rgb1 = points_input1[:,3:6]
    xyz_input2 = points_input2[:,:3]
    rgb2 = points_input2[:,3:6]
    if trans is not None:
        # convert points to ball space
        xyz_input1 = (xyz_input1 - trans[1:]) / trans[0]
        xyz_input2 = (xyz_input2 - trans[1:]) / trans[0]

    if flownet is not None:
        with torch.no_grad():
            input1 = torch.tensor(xyz_input1).unsqueeze(0).transpose(1,2).contiguous().to("cuda:0")
            input2 = torch.tensor(xyz_input2).unsqueeze(0).transpose(1,2).contiguous().to("cuda:0")
            feat1 = torch.tensor(rgb1).unsqueeze(0).transpose(1,2).contiguous().to("cuda:0")
            feat2 = torch.tensor(rgb2).unsqueeze(0).transpose(1,2).contiguous().to("cuda:0")
            output = flownet(input1, input2, feat1, feat2)
            visu_flows_with_start_points(output[:,:,0:2048], input1[:,:,0:2048])
    else:
        # use icp cpu instead
        indices = icp_mask(torch.cat([torch.tensor(xyz_input1), torch.tensor(rgb1)], dim=1).unsqueeze(0), torch.cat([torch.tensor(xyz_input2), torch.tensor(rgb2)], dim=1).unsqueeze(0), directly_nn)
        output = (torch.tensor(xyz_input2).unsqueeze(0)[:,indices[0],:] - torch.tensor(xyz_input1).unsqueeze(0)).permute(0,2,1)
        visu_flows_with_start_points(output[:,:,0:2048], torch.tensor(xyz_input1).unsqueeze(0).permute(0,2,1)[:,:,0:2048])

    if not use_origin_color:
        # use red for image 1 and blue for image 2
        rgb1[:,0:3] = np.array([1, 0, 0])  # red
        rgb2[:,0:3] = np.array([0, 0, 1])  # blue
    if add_flow:
        xyz_input1 = xyz_input1 + output[0,0:3,:].permute(1,0).cpu().numpy()
    if trans is not None:
        # add it back for draw graph
        xyz_input1 = xyz_input1 * trans[0] + trans[1:]
        xyz_input2 = xyz_input2 * trans[0] + trans[1:]
    xyz_world = np.concatenate((xyz_input1, xyz_input2), axis=0)
    rgb = np.concatenate((rgb1, rgb2), axis=0)
    pc_img_world = map2image(xyz_world, rgb*255.0)
    pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
    plt.imshow(pc_img_world)