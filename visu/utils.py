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
            break

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
            break

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


def save_part_point_cloud(xyz, colors, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, alpha=0.5)
    ax.set_axis_off()
    plt.savefig(save_path)
    plt.close(fig)

def draw_vector_on_3d_plot(ax, center, vector, color, scale=1.0, arrow_length_ratio=0.1, linewidth=2):
    start_point = center
    end_point = center + vector * scale
    ax.quiver(start_point[0], start_point[1], start_point[2], 
              vector[0], vector[1], vector[2], 
              color=color, length=scale, normalize=True, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)

def random_generate_rotation_matrix(min, max):
    x = random.uniform(min, max)
    y = random.uniform(min, max)
    z = random.uniform(min, max)

    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])

    R = Rz @ Ry @ Rx
    return R.astype(np.float32)