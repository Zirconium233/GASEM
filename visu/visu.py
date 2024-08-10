import torch
import numpy as np
import yaml
from os.path import join as pjoin
import os
import argparse
import importlib
from datasets.GAPartNet.dataset.point_cloud import PointCloud
from datasets.GAPartNet.misc.pose_fitting import estimate_pose_from_npcs
import cv2
from typing import List, Dict
import glob
import json
from network.utils import vectors_to_rotation_matrix, calculate_pose_metrics, random_generate_rotation_matrix
from datasets.GAPartNet.misc.visu_util import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 
from network.utils import vectors_to_rotation_matrix, get_gt_v
import matplotlib.pyplot as plt
import copy
import random
from visu.utils import draw_vector_on_3d_plot, view_pc1_to_pc2, save_part_point_cloud, visu_flows, visu_flows_with_start_points
from network.icp import icp, icp_gpu, icp_mask

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

def draw_gpv(GAPARTNET_DATA_ROOT, dataset, 
             gpv_net = None, i: int = None, 
             draw_vector = True, save_root = None, 
             flip=[0,0,0], view = None, scale=1.0, arrow_length_ratio=0.1, linewidth=2,
             net = None, rot = True, show = True):
    if i is None:
        i = random.randint(0, len(dataset))
    if gpv_net is None:
        print("use ground truth")
    print('index: ', i)
    name = f"visu_{i}"
    inputs = dataset[i]
    name_1 = dataset.group_files[i][0].split('/')[-1].split('.')[0]
    name_2 = dataset.group_files[i][1].split('/')[-1].split('.')[0]
    print("names: ", name_1, " and ", name_2)
    points_input1, meta_all1, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_1, False)
    points_input2, meta_all2, _, __, ___ = process_gapartnetfile(GAPARTNET_DATA_ROOT, name_2, False)
    
    points_input1_ = points_input1.numpy()
    points_input2_ = points_input2.numpy()
    
    xyz_input1 = points_input1_[:,:3]
    rgb1 = points_input1_[:,3:6]
    trans1 = meta_all1['scale_param']
    xyz_input2 = points_input2_[:,:3]
    rgb2 = points_input2_[:,3:6]
    trans2 = meta_all2['scale_param']

    # xyz_input1 = visu.rotation_to_world(xyz_input1, inputs.rot_1)
    # xyz_input2 = visu.rotation_to_world(xyz_input2, inputs.rot_2)

    xyz1 = xyz_input1 * trans1[0] + trans1[1:4]
    xyz2 = xyz_input2 * trans2[0] + trans2[1:4]

    xyz_input1_ = points_input1_[:,:3]
    # rgb1_ = points_input1_[:,3:6]
    xyz_input2_ = points_input2_[:,:3]
    # rgb2_ = points_input2_[:,3:6]
    if gpv_net is None:
        xyz_input1_ = rotation_to_world(xyz_input1_, torch.tensor(meta_all1["world2camera_rotation"]).reshape(3, 3))
        xyz_input2_ = rotation_to_world(xyz_input2_, torch.tensor(meta_all2["world2camera_rotation"]).reshape(3, 3))
    else:
        (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = gpv_net([inputs.to('cuda:0')])
        rot_1_pred = vectors_to_rotation_matrix(p_green_R1.detach(), p_red_R1.detach(), True)
        rot_2_pred = vectors_to_rotation_matrix(p_green_R2.detach(), p_red_R2.detach(), True)
        error1 = calculate_pose_metrics(rot_1_pred, torch.stack([inputs.rot_1.transpose(0,1)]))
        print("error1: ", error1)
        error2 = calculate_pose_metrics(rot_2_pred, torch.stack([inputs.rot_2.transpose(0,1)]))
        print("error2: ", error2)
        # xyz_input1_ = visu.rotation_to_world(xyz_input1_, rot_1_pred[0].transpose(0,1).cpu())
        xyz_input2_ = rotation_to_world(xyz_input2_, (rot_2_pred[0] @ rot_1_pred[0].transpose(0,1)).transpose(0,1).cpu())
    xyz1_ = xyz_input1_ * trans1[0] + trans1[1:4]
    xyz2_ = xyz_input2_ * trans2[0] + trans2[1:4]

    fig = plt.figure(figsize=(12, 18))
    
    
    
    ax1 = fig.add_subplot(3, 2, 1, projection='3d')
    ax1.scatter(xyz1[:2000, 0], xyz1[:2000, 1], xyz1[:2000, 2], c=rgb1[:2000,:], alpha=0.5)
    if draw_vector:
        center1 = trans1[1:4]
        draw_vector_on_3d_plot(ax1, center1, p_green_R1[0].detach().cpu().numpy(), 'g', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        draw_vector_on_3d_plot(ax1, center1, p_red_R1[0].detach().cpu().numpy(), 'r', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        p_blue_R1 = np.cross(p_green_R1[0].detach().cpu().numpy(), p_red_R1[0].detach().cpu().numpy())
        draw_vector_on_3d_plot(ax1, center1, p_blue_R1, 'b', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
    ax1.set_title('Original Image 1')
    ax1.set_axis_off()

    ax2 = fig.add_subplot(3, 2, 2, projection='3d')
    ###
    # random_rot = random_generate_rotation_matrix(0, np.pi * 1/4)
    # xyz2 = visu.rotation_to_world(xyz_input2, random_rot) * trans2[0] + trans2[1:4]
    ###
    ax2.scatter(xyz2[:2000, 0], xyz2[:2000, 1], xyz2[:2000, 2], c=rgb2[:2000,:], alpha=0.5)
    if draw_vector:
        center2 = trans2[1:4]
        p_green_R2_n = p_green_R2[0].detach().cpu().numpy()
        p_red_R2_n = p_red_R2[0].detach().cpu().numpy()
        ###
        # p_green_R2_n = random_rot @ p_green_R2_n
        # p_red_R2_n = random_rot @ p_red_R2_n
        ###
        draw_vector_on_3d_plot(ax2, center2, p_green_R2_n, 'g', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        draw_vector_on_3d_plot(ax2, center2, p_red_R2_n, 'r', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        p_blue_R2 = np.cross(p_green_R2_n, p_red_R2_n)
        draw_vector_on_3d_plot(ax2, center2, p_blue_R2, 'b', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
    ax2.set_title('Original Image 2')
    ax2.set_axis_off()

    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    ax3.scatter(xyz1_[:2000, 0], xyz1_[:2000, 1], xyz1_[:2000, 2], c=rgb1[:2000,:], alpha=0.5)
    if draw_vector:
        center1 = trans1[1:4]
        # p_green_R1_ = rot_1_pred[0].transpose(0,1).cpu().numpy() @ p_green_R1[0].detach().cpu().numpy()
        # p_red_R1_ = rot_1_pred[0].transpose(0,1).cpu().numpy() @ p_red_R1[0].detach().cpu().numpy()
        # only rotate the vector of point cloud 2
        p_green_R1_ = p_green_R1[0].detach().cpu().numpy()
        p_red_R1_ = p_red_R1[0].detach().cpu().numpy()
        draw_vector_on_3d_plot(ax3, center1, p_green_R1_, 'g', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        draw_vector_on_3d_plot(ax3, center1, p_red_R1_, 'r', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        p_blue_R1 = np.cross(p_green_R1_, p_red_R1_)
        draw_vector_on_3d_plot(ax3, center1, p_blue_R1, 'b', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
    ax3.set_title('Transformed Image 1')
    ax3.set_axis_off()

    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    ax4.scatter(xyz2_[:2000, 0], xyz2_[:2000, 1], xyz2_[:2000, 2], c=rgb2[:2000,:], alpha=0.5)
    if draw_vector:
        center2 = trans2[1:4]
        # rotate to point cloud 1's pose
        p_green_R2_ = rot_1_pred[0].detach().cpu().numpy() @ rot_2_pred[0].transpose(0,1).cpu().numpy() @ p_green_R2[0].detach().cpu().numpy()
        p_red_R2_ = rot_1_pred[0].detach().cpu().numpy() @ rot_2_pred[0].transpose(0,1).cpu().numpy() @ p_red_R2[0].detach().cpu().numpy()
        draw_vector_on_3d_plot(ax4, center2, p_green_R2_, 'g', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        draw_vector_on_3d_plot(ax4, center2, p_red_R2_, 'r', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
        p_blue_R2 = np.cross(p_green_R2_, p_red_R2_)
        draw_vector_on_3d_plot(ax4, center2, p_blue_R2, 'b', scale=scale, arrow_length_ratio=arrow_length_ratio, linewidth=linewidth)
    ax4.set_title('Transformed Image 2')
    ax4.set_axis_off()

    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    ax5.scatter(xyz1_[:2000, 0], xyz1_[:2000, 1], xyz1_[:2000, 2], c=rgb1[:2000,:], alpha=0.5)
    # draw flows
    if rot:
        indices = icp_mask(torch.cat([torch.tensor(xyz1_), torch.tensor(rgb1)], dim=1).unsqueeze(0), torch.cat([torch.tensor(xyz2_), torch.tensor(rgb2)], dim=1).unsqueeze(0), True)
        flows = (torch.tensor(xyz_input2).unsqueeze(0)[:,indices[0],:] - torch.tensor(xyz_input1).unsqueeze(0))[0].numpy() # (20000, 6)
    else:
        indices = icp_mask(torch.cat([torch.tensor(points_input1_), torch.tensor(rgb1)], dim=1).unsqueeze(0), torch.cat([torch.tensor(points_input2_), torch.tensor(rgb2)], dim=1).unsqueeze(0), True)
        flows = (torch.tensor(points_input2_).unsqueeze(0)[:,indices[0],:] - torch.tensor(points_input1_).unsqueeze(0))[0].numpy() # (20000, 6)

    ax5.quiver(
        xyz1_[:500, 0], xyz1_[:500, 1], xyz1_[:500, 2], 
        flows[:500, 0], flows[:500, 1], flows[:500, 2], 
        color='r', normalize=False, arrow_length_ratio=0.3, linewidth=2, length=0.5
    )
    ax5.set_title('Image1 with Flows')
    ax5.set_axis_off()

    ax6 = fig.add_subplot(3, 2, 6, projection='3d')
    if net is None:
    # set the color of image 1 to red and 2 to blue
        rgb1_ = copy.copy(rgb1)
        rgb1_[:,0:3] = [1, 0, 0]
        rgb2_ = copy.copy(rgb2)
        rgb2_[:,0:3] = [0, 0, 1]
        ax6.scatter(np.concatenate([xyz2_[:500, 0], xyz1_[:500, 0]]), np.concatenate([xyz2_[:500, 1], xyz1_[:500, 1]]), np.concatenate([xyz2_[:500, 2], xyz1_[:500, 2]]), c=np.concatenate([rgb2_[:500,:], rgb1_[:500,:]]), alpha=0.5)
        ax6.set_title('Concatenated Image')
    else:
        rgb1_ = copy.copy(rgb1)
        rgb2_ = copy.copy(rgb2)
        pc_pair_ = copy.deepcopy(inputs.to('cpu')).to('cuda:0')
        if rot:
            pc_pair_.pc1.points[:,0:3] = (rot_1_pred[0].transpose(0,1) @ pc_pair_.pc1.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous()
            pc_pair_.pc2.points[:,0:3] = (rot_2_pred[0].transpose(0,1) @ pc_pair_.pc2.points[:,0:3].transpose(0,1)).transpose(0,1).contiguous()
            flow = None
        else:
            flow = torch.tensor(flows[:,0:3], dtype=torch.float32).unsqueeze(0).permute(0,2,1).cuda()
            flow = None

        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets = forward_model(net, [pc_pair_], flow_data=flow, do_inference=True)
        # rotate the bbox points back bbox size is [1,n,8,3]
        if rot:
            bboxes = (np.asarray(bboxes) @ rot_1_pred[0].cpu().numpy().T).tolist()
        proposal_indices = proposal_indices.detach().cpu().numpy()
        proposal_offsets = proposal_offsets.detach().cpu().numpy()
        ins_pred_color = np.ones_like(rgb1_) * 230 / 255.0
        for ins_i in range(len(proposal_offsets) - 1):
            ins_pred_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = COLOR20[ins_i % 19 + 1] / 255.0

            part_mask = np.zeros_like(rgb1_)
            part_mask[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = 1.0
            part_color = part_mask * COLOR20[ins_i % 19 + 1] / 255.0
            part_xyz = xyz1_[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]]
            save_path = os.path.join(save_root, name, f'part_{ins_i}.png')
            save_part_point_cloud(part_xyz, part_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]], save_path)
        ax6.scatter(xyz1_[:2000, 0], xyz1_[:2000, 1], xyz1_[:2000, 2], c=ins_pred_color[:2000,:], alpha=0.5)
        ax6.set_title('Instance Segmentation')
    ax6.set_axis_off()
    
    if flip[0]:
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax2.set_xlim(ax2.get_xlim()[::-1])
        ax3.set_xlim(ax3.get_xlim()[::-1])
        ax4.set_xlim(ax4.get_xlim()[::-1])
        ax5.set_xlim(ax5.get_xlim()[::-1])
        ax6.set_xlim(ax6.get_xlim()[::-1])
    if flip[1]:
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax3.set_ylim(ax3.get_ylim()[::-1])
        ax4.set_ylim(ax4.get_ylim()[::-1])
        ax5.set_ylim(ax5.get_ylim()[::-1])
        ax6.set_ylim(ax6.get_ylim()[::-1])
    if flip[2]:
        ax1.set_zlim(ax1.get_zlim()[::-1])
        ax2.set_zlim(ax2.get_zlim()[::-1])
        ax3.set_zlim(ax3.get_zlim()[::-1])
        ax4.set_zlim(ax4.get_zlim()[::-1])
        ax5.set_zlim(ax5.get_zlim()[::-1])
        ax6.set_zlim(ax6.get_zlim()[::-1])
    if view is not None:
        ax1.view_init(view[0], view[1])
        ax2.view_init(view[0], view[1])
        ax3.view_init(view[0], view[1])
        ax4.view_init(view[0], view[1])
        ax5.view_init(view[0], view[1])
        ax6.view_init(view[0], view[1])

    if save_root is not None:
        plt.savefig(save_root + '/' + name + '/' + 'rot_and_gt_v2.png', format='png')
    if show:
        plt.show()
    return rot_1_pred, rot_2_pred, flow

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

def random_show_flow_pc1_to_pc2(gpv_net, flownet, dataset, dir_name: Dict, use_gt=False, use_icp=False, idx=None, icp_device="gpu"):
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

def forward_model(gapnet, points_list, flow_data=None, do_inference=True):
    """
    Args:
        gapnet: the network to forward
        points_list: a list of points, each element is a point cloud pair
        flow_data: the flow data, if not None, the network will use the flow data to warp the points. [bs, 3, num_points]
        do_inference: if True, the network will return the output of the network, otherwise, the network will return the output of the network and the intermediate results
    Returns:
        bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets
    """
    device = gapnet.device
    with torch.no_grad():
        out_dict = gapnet(points_list, flow_data=flow_data, do_inference=do_inference)
    print(out_dict['loss_sem_seg'])
    sem_preds = out_dict['sem_preds']
    proposals = out_dict['proposals']
    if proposals is not None:
        pt_xyz = proposals.pt_xyz
        batch_indices = proposals.batch_indices
        proposal_offsets = proposals.proposal_offsets
        num_points_per_proposal = proposals.num_points_per_proposal
        num_proposals = num_points_per_proposal.shape[0]
        npcs_preds = proposals.npcs_preds
        score_preds= proposals.score_preds

    indices = torch.arange(sem_preds.shape[0], dtype=torch.int64, device="cuda:0")
    proposal_indices = indices[proposals.valid_mask][proposals.sorted_indices]
    npcs_maps = points_list[0].pc1.points[:,:3].clone().to("cuda:0")
    npcs_maps[:] = 230./255.
    if proposals is not None:
        npcs_maps[proposal_indices] = npcs_preds
    bboxes = [[] for _ in range(len(points_list))]
    if proposals is not None:
        for i in range(num_proposals):
            offset_begin = proposal_offsets[i].item()
            offset_end = proposal_offsets[i + 1].item()

            batch_idx = batch_indices[offset_begin]
            xyz_i = pt_xyz[offset_begin:offset_end]
            npcs_i = npcs_preds[offset_begin:offset_end]

            npcs_i = npcs_i - 0.5
            if xyz_i.shape[0]<=4:
                continue
            bbox_xyz, scale, rotation, translation, out_transform, best_inlier_idx = \
                estimate_pose_from_npcs(xyz_i.cpu().numpy(), npcs_i.cpu().numpy())
            # import pdb
            # pdb.set_trace()
            if scale[0] == None:
                continue
            bboxes[batch_idx].append(bbox_xyz.tolist())
    try:
        return bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets
    except:
        print("no proposal")
        return bboxes, sem_preds, npcs_maps, None, None
    
def draw_result(save_option, save_root, name, points_input, trans, bboxes, sem_preds, npcs_maps, proposal_indices, proposal_offsets, gts=None, have_proposal = True, save_local = False):
    
    final_save_root = f"{save_root}/"
    save_root = f"{save_root}/{name}/"
    if save_local:
        os.makedirs(save_root, exist_ok=True)
    final_img = np.ones((3 * (HEIGHT + EDGE) + EDGE, 4 * (WIDTH + EDGE) + EDGE, 3), dtype=np.uint8) * 255
    xyz_input = points_input[:,:3]
    rgb = points_input[:,3:6]
    xyz = xyz_input * trans[0] + trans[1:4]
    pc_img = map2image(xyz, rgb*255.0)
    pc_img = cv2.cvtColor(pc_img, cv2.COLOR_BGR2RGB)
    # if "raw" in save_option:
    #     raw_img_path = f"{RAW_IMG_ROOT}/{name}.png"
    #     if os.path.exists(raw_img_path):
    #         raw_img = cv2.imread(raw_img_path)
    #         if save_local:
    #             cv2.imwrite(f"{save_root}/raw.png", raw_img)
    #         X_START = EDGE
    #         Y_START = EDGE
    #         final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = raw_img
    #         text = "raw"
    #         cv2.putText(final_img, text, 
    #             (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
    #             font, fontScale, fontColor, thickness, lineType)
    if "pc" in save_option:
        if save_local:
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
        if save_local:
            cv2.imwrite(f"{save_root}/sem_pred.png", sem_pred_img)
        X_START = EDGE + (WIDTH + EDGE)
        Y_START = EDGE + (HEIGHT + EDGE)
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_pred_img
        text = "sem_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_pred" in save_option:
        ins_pred_color = np.ones_like(xyz) * 230
        if have_proposal:
            for ins_i in range(len(proposal_offsets) - 1):
                ins_pred_color[proposal_indices[proposal_offsets[ins_i]:proposal_offsets[ins_i + 1]]] = COLOR20[ins_i%19 + 1]
        
        ins_pred_img = map2image(xyz, ins_pred_color)
        ins_pred_img = cv2.cvtColor(ins_pred_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/ins_pred.png", ins_pred_img)    
        X_START = EDGE + (WIDTH + EDGE) * 1
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_pred_img
        text = "ins_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_pred" in save_option:
        npcs_pred_img = map2image(xyz, npcs_maps*255.0)
        npcs_pred_img = cv2.cvtColor(npcs_pred_img, cv2.COLOR_BGR2RGB)
        if save_local:
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
        img_bbox_pred = pc_img.copy()
        draw_bbox(img_bbox_pred, bboxes[0], trans)
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_pred.png", img_bbox_pred)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_pred
        text = "bbox_pred"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "pure_bbox" in save_option:
        img_empty = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255
        draw_bbox(img_empty, bboxes[0], trans)
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_pure.png", img_empty)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 3
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_empty
        text = "bbox_pred_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "sem_gt" in save_option:
        sem_gt = gts[0]
        sem_gt_img = map2image(xyz, COLOR20[sem_gt])
        sem_gt_img = cv2.cvtColor(sem_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/sem_gt.png", sem_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 1
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = sem_gt_img
        text = "sem_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "ins_gt" in save_option:
        ins_gt = gts[1]
        ins_color = COLOR20[ins_gt%19 + 1]
        ins_color[np.where(ins_gt == -100)] = 230
        ins_gt_img = map2image(xyz, ins_color)
        
        ins_gt_img = cv2.cvtColor(ins_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
            cv2.imwrite(f"{save_root}/ins_gt.png", ins_gt_img)      
        X_START = EDGE + (WIDTH + EDGE) * 0
        Y_START = EDGE + (HEIGHT + EDGE) * 2
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = ins_gt_img
        text = "ins_gt"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    if "npcs_gt" in save_option:
        npcs_gt = gts[2] + 0.5
        npcs_gt_img = map2image(xyz, npcs_gt*255.0)
        npcs_gt_img = cv2.cvtColor(npcs_gt_img, cv2.COLOR_BGR2RGB)
        if save_local:
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
        ins_gt = gts[1]
        npcs_gt = gts[2]
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
        if save_local:
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
        ins_gt = gts[1]
        npcs_gt = gts[2]
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
        if save_local:
            cv2.imwrite(f"{save_root}/bbox_gt_pure.png", img_bbox_gt_pure)
        X_START = EDGE + (WIDTH + EDGE) * 2
        Y_START = EDGE + (HEIGHT + EDGE) * 0
        final_img[X_START:X_START+HEIGHT, Y_START:Y_START+WIDTH, :] = img_bbox_gt_pure
        text = "bbox_gt_pure"
        cv2.putText(final_img, text, 
            (Y_START + int(0.5*(WIDTH - 3 * EDGE)), X_START + HEIGHT + int(0.5*EDGE)), 
            font, fontScale, fontColor, thickness, lineType)
    cv2.imwrite(f"{final_save_root}/{name}.png", final_img)

if __name__ == "__main__":
    save_root = "./log_dir/visu"
    gap_root = "./datasets/GAPartNet/dataset/data/"

    
    