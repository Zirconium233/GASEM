import torch
import torch.nn as nn
from typing import List, Dict, Union
from torch.utils.data import Dataset
from network.utils import vectors_to_rotation_matrix, get_gt_v
from datasets.GAPartNet.misc.pose_fitting import estimate_pose_from_npcs
from loss.utils import calculate_pose_metrics, calculate_pose_metrics_quaternion
import random
import cv2
import numpy as np
from datasets.GAPartNet.misc.visu_util import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 
import matplotlib.pyplot as plt
from network.icp import icp, icp_gpu, icp_mask
import os
import json

def camera_to_world(xyz, meta_all):
    camera_intrinsic = np.array(meta_all["camera_intrinsic"]).reshape(3, 3)
    world2camera_rotation = np.array(meta_all["world2camera_rotation"]).reshape(3, 3)
    camera2world_translation = np.array(meta_all["camera2world_translation"])

    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = world2camera_rotation
    extrinsic_matrix[:3, 3] = -camera2world_translation

    xyz_camera_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

    transform_matrix = np.linalg.inv(extrinsic_matrix)

    xyz_world_homogeneous = np.dot(transform_matrix, xyz_camera_homogeneous.T).T

    xyz_world = xyz_world_homogeneous[:, :3]

    return xyz_world

def rotation_to_world_tensor(xyz_input: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    world2camera_rotation = rot
    xyz_rotated = torch.bmm(world2camera_rotation, xyz_input.T)
    return xyz_rotated.T

def rotation_to_world(xyz_input: np.ndarray, rot: np.ndarray) -> np.ndarray:
    # world2camera_rotation = np.array(meta_all["world2camera_rotation"]).reshape(3, 3)
    world2camera_rotation = rot
    xyz_rotated = np.dot(world2camera_rotation, xyz_input.T)
    return xyz_rotated.T

def process_gapartnetfile(GAPARTNET_DATA_ROOT, name, five):
    data_path = f"{GAPARTNET_DATA_ROOT}/pth/{name}.pth"
    trans = None
    if five:
        trans_path = f"{GAPARTNET_DATA_ROOT}/meta/{name}.txt"
        pc, rgb, semantic_label, instance_label, npcs_map = torch.load(data_path)
        trans = np.loadtxt(trans_path)
    else:
        trans_path = f"{GAPARTNET_DATA_ROOT}/meta/{name}.json"
        pc, rgb, semantic_label, instance_label, npcs_map, idx = torch.load(data_path)
        meta_all = None
        with open(trans_path, 'r+', encoding='utf-8') as f:
            meta_all = json.loads(f.read())
        trans = meta_all['scale_param']
    xyz = pc * trans[0] + trans[1:4]

    # save_point_cloud_to_ply(xyz, rgb*255, data_path.split("/")[-1].split(".")[0]+"_preinput.ply")
    # save_point_cloud_to_ply(pc, rgb*255, data_path.split("/")[-1].split(".")[0]+"_input.ply")
    points_input = torch.cat((torch.tensor(pc),torch.tensor(rgb)), dim = 1)
    if five:
        return points_input, trans, semantic_label, instance_label, npcs_map
    else:
        return points_input, meta_all, semantic_label, instance_label, npcs_map

def visualize_gapartnet(
    SAVE_ROOT, 
    GAPARTNET_DATA_ROOT,
    RAW_IMG_ROOT,
    save_option: List = [], 
    name: str = "pc",
    bboxes: np.ndarray = None, # type: ignore
    sem_preds: np.ndarray = None, # type: ignore
    ins_preds: np.ndarray = None, # type: ignore
    npcs_preds: np.ndarray = None, # type: ignore
    rot_pred: np.ndarray = None, # type: ignore
    have_proposal = True, 
    save_detail = False,
    five = False,
):
    
    final_save_root = f"{SAVE_ROOT}"
    save_root = f"{SAVE_ROOT}/{name}"
    os.makedirs(final_save_root, exist_ok=True)
    if save_detail:
        os.makedirs(f"{save_root}", exist_ok=True)
    final_img = np.ones((3 * (HEIGHT + EDGE) + EDGE, 4 * (WIDTH + EDGE) + EDGE, 3), dtype=np.uint8) * 255
    
    points_input, meta_all, semantic_label, instance_label, npcs_map = process_gapartnetfile(GAPARTNET_DATA_ROOT, name, five)

    points_input = points_input.numpy()
    xyz_input = points_input[:,:3]
    rgb = points_input[:,3:6]
    if not five:
        trans = meta_all['scale_param']
    else:
        trans = meta_all
    xyz = xyz_input * trans[0] + trans[1:4]
    pc_img = map2image(xyz, rgb*255.0)
    pc_img = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
    
    if "raw" in save_option:
        raw_img_path = f"{RAW_IMG_ROOT}/{name}.png"
        if os.path.exists(raw_img_path):
            raw_img = cv2.imread(raw_img_path)
            if save_detail:
                cv2.imwrite(f"{save_root}/raw.png", raw_img)
            X_START = EDGE
            Y_START = EDGE
            final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = raw_img
            text = "raw"
            cv2.putText(final_img, text, 
                (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
                font, fontScale, fontColor, thickness, lineType)
    if "pc" in save_option:
        if save_detail:
            cv2.imwrite(f"{save_root}/pc.png", pc_img)
        X_START = EDGE + (HEIGHT + EDGE)
        Y_START = EDGE
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = pc_img
        text = "pc"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_pred" in save_option:
        sem_pred_img = map2image(xyz, COLOR20[sem_preds])
        sem_pred_img = cv2.cvtColor(sem_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/sem_pred.png", sem_pred_img)
        X_START = EDGE + (WIDTH + EDGE)
        Y_START = EDGE + (HEIGHT + EDGE)
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_pred_img
        text = "sem_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_pred" in save_option:
        # ins_pred_color = np.ones_like(xyz) * 230
        # if have_proposal:
        #     for ins_i in range(len(proposal_offsets) - 1):
        #         ins_pred_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = COLOR20[ins_i%19 + 1]
        # import pdb; pdb.set_trace()
        ins_pred_img = map2image(xyz, COLOR20[(ins_preds%20).astype(np.int_)])
        ins_pred_img = cv2.cvtColor(ins_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/ins_pred.png", ins_pred_img)    
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_pred_img
        text = "ins_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_pred" in save_option:
        npcs_pred_img = map2image(xyz, npcs_preds*255.0)
        npcs_pred_img = cv2.cvtColor(npcs_pred_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/npcs_pred.png", npcs_pred_img)
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        # import pdb
        # pdb.set_trace()
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = npcs_pred_img
        text = "npcs_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_pred" in save_option:
        assert "world_pred" not in save_option, "position conflicted"
        img_bbox_pred = pc_img.copy()
        draw_bbox(img_bbox_pred, bboxes, trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_pred.png", img_bbox_pred)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_pred
        text = "bbox_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_pred_pure" in save_option:
        assert "world_gt" not in save_option, "position conflicted"
        img_empty = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_empty, bboxes, trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_pure.png", img_empty)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_empty
        text = "bbox_pred_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_gt" in save_option:
        sem_gt = semantic_label
        sem_gt_img = map2image(xyz, COLOR20[sem_gt])
        sem_gt_img = cv2.cvtColor(sem_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/sem_gt.png", sem_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_gt_img
        text = "sem_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_gt" in save_option:
        ins_gt = instance_label
        ins_color = COLOR20[ins_gt%19 + 1]
        ins_color[np.where(ins_gt == -100)] = 230
        ins_gt_img = map2image(xyz, ins_color)
        
        ins_gt_img = cv2.cvtColor(ins_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/ins_gt.png", ins_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_gt_img
        text = "ins_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_gt" in save_option:
        npcs_gt = npcs_map + 0.5
        npcs_gt_img = map2image(xyz, npcs_gt*255.0)
        npcs_gt_img = cv2.cvtColor(npcs_gt_img, cv2.COLOR_BGR2RGB)
        if save_detail:
            cv2.imwrite(f"{save_root}/npcs_gt.png", npcs_gt_img)
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = npcs_gt_img
        text = "npcs_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_gt" in save_option:
        bboxes_gt = [[]]
        ins_gt = instance_label
        npcs_gt = npcs_map
        # import pdb
        # pdb.set_trace()
        num_ins = ins_gt.max()+1
        if num_ins >= 1:
            for ins_i in range(num_ins):
                mask_i = ins_gt == ins_i
                xyz_input_i = xyz_input[mask_i]
                npcs_i = npcs_gt[mask_i]
                if xyz_input_i.shape[0]<=5:
                    continue

                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                    estimate_pose_from_npcs(xyz_input_i, npcs_i)
                if scale[0] == None:
                    continue
                bboxes_gt[0].append(bbox_xyz.tolist())
        img_bbox_gt = pc_img.copy()
        draw_bbox(img_bbox_gt, bboxes_gt[0], trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_gt.png", img_bbox_gt)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt
        text = "bbox_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "bbox_gt_pure" in save_option:
        bboxes_gt = [[]]
        ins_gt = instance_label
        npcs_gt = npcs_map
        # import pdb
        # pdb.set_trace()
        num_ins = ins_gt.max()+1
        if num_ins >= 1:
            for ins_i in range(num_ins):
                mask_i = ins_gt == ins_i
                xyz_input_i = xyz_input[mask_i]
                npcs_i = npcs_gt[mask_i]
                if xyz_input_i.shape[0]<=5:
                    continue

                bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                    estimate_pose_from_npcs(xyz_input_i, npcs_i)
                if scale[0] == None:
                    continue

                bboxes_gt[0].append(bbox_xyz.tolist())
        img_bbox_gt_pure = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_bbox_gt_pure, bboxes_gt[0], trans)
        if save_detail:
            cv2.imwrite(f"{save_root}/bbox_gt_pure.png", img_bbox_gt_pure)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 0
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt_pure
        text = "bbox_gt_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "world_gt" in save_option:
        if five:
            print("no rotation to use. ")
        else:
            assert "bbox_pred_pure" not in save_option, "position conflicted"
            xyz_world_ball = rotation_to_world(xyz_input, np.array(meta_all["world2camera_rotation"]).reshape(3, 3))
            xyz_world = xyz_world_ball * trans[0] + trans[1:4]
            pc_img_world = map2image(xyz_world, rgb*255.0)
            pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
            if save_detail:
                cv2.imwrite(f"{save_root}/world_gt.png", pc_img_world)
            X_START = EDGE + (WIDTH + EDGE) * 2
            Y_START = EDGE + (HEIGHT + EDGE) * 3
            final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = pc_img_world
            text = "world_gt"
            cv2.putText(final_img, text, 
                (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
                font, fontScale, fontColor, thickness, lineType)
    if "world_pred" in save_option:
        if five:
            print("no rotation to use. ")
        else:
            assert "bbox_pred_pure" not in save_option, "position conflicted"
            xyz_world_ball = rotation_to_world(xyz_input, rot_pred)
            xyz_world = xyz_world_ball * trans[0] + trans[1:4]
            pc_img_world = map2image(xyz_world, rgb*255.0)
            pc_img_world = cv2.cvtColor(pc_img_world, cv2.COLOR_BGR2RGB)
            if save_detail:
                cv2.imwrite(f"{save_root}/world_pred.png", pc_img_world)
            X_START = EDGE + (WIDTH + EDGE) * 1
            Y_START = EDGE + (HEIGHT + EDGE) * 3
            final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = pc_img_world
            text = "world_pred"
            cv2.putText(final_img, text, 
                (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
                font, fontScale, fontColor, thickness, lineType)
    cv2.imwrite(f"{final_save_root}/{name}.png", final_img)

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
        visualize_gapartnet(f"./log_dir/GPV_test/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_1, rot_pred = rot_1_pred.detach().squeeze(0).cpu(), five=False)
        visualize_gapartnet(f"./log_dir/GPV_test/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_2, rot_pred = rot_2_pred.detach().squeeze(0).cpu(), five=False)
    
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
        visualize_gapartnet(f"./log_dir/GPV_test_sym_v1/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_1, rot_pred = rot_1_pred.detach().squeeze(0).cpu(), five=False)
        visualize_gapartnet(f"./log_dir/GPV_test_sym_v1/visu/{log_name[dataset]}", dir_name[dataset], None, ['pc', 'world_gt', 'world_pred'], name_2, rot_pred = rot_2_pred.detach().squeeze(0).cpu(), five=False)
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
    
    points_input1, meta_all1, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

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

    xyz_world_ball1 = rotation_to_world(xyz_input1, np.array(meta_all1["world2camera_rotation"]).reshape(3, 3))
    xyz_world_ball2 = rotation_to_world(xyz_input2, np.array(meta_all2["world2camera_rotation"]).reshape(3, 3))
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
    points_input1, meta_all1, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

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
    xyz_world_ball1 = rotation_to_world(xyz_input1, rot1)
    xyz_world_ball2 = rotation_to_world(xyz_input2, rot2)

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
    points_input1, meta_all1, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)
    
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
        xyz_input1 = rotation_to_world(xyz_input1, inputs.rot_1)
        xyz_input2 = rotation_to_world(xyz_input2, inputs.rot_2)

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
        xyz_input1_ = rotation_to_world(xyz_input1_, torch.tensor(meta_all1["world2camera_rotation"]).reshape(3, 3))
        xyz_input2_ = rotation_to_world(xyz_input2_, torch.tensor(meta_all2["world2camera_rotation"]).reshape(3, 3))

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
    points_input1, meta_all1, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)

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

    xyz_world_ball1 = rotation_to_world(xyz_input1, rot1)
    xyz_world_ball2 = rotation_to_world(xyz_input2, rot2)

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