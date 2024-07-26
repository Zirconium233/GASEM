import torch
import numpy as np
import yaml
from os.path import join as pjoin
import os
import argparse
import sys
sys.path.append(sys.path[0] + "/..")
import importlib
from datasets.GAPartNet.dataset.point_cloud import PointCloud
from datasets.GAPartNet.misc.pose_fitting import estimate_pose_from_npcs
import cv2
from typing import List
import glob
import json
from datasets.GAPartNet.misc.visu_util import OBJfile2points, map2image, save_point_cloud_to_ply, \
    WorldSpaceToBallSpace, FindMaxDis, draw_bbox_old, draw_bbox, COLOR20, \
    OTHER_COLOR, HEIGHT, WIDTH, EDGE, K, font, fontScale, fontColor,thickness, lineType 

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
    # 提取相机内参和外参
    camera_intrinsic = np.array(meta_all["camera_intrinsic"]).reshape(3, 3)
    world2camera_rotation = np.array(meta_all["world2camera_rotation"]).reshape(3, 3)
    camera2world_translation = np.array(meta_all["camera2world_translation"])

    # 组合外参矩阵
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = world2camera_rotation
    extrinsic_matrix[:3, 3] = -camera2world_translation

    # 将相机坐标系下的点坐标转换为齐次坐标
    xyz_camera_homogeneous = np.hstack((xyz, np.ones((xyz.shape[0], 1))))

    # 相机坐标系到世界坐标系的变换
    transform_matrix = np.linalg.inv(extrinsic_matrix)

    # 将点坐标从相机坐标系转换为世界坐标系
    xyz_world_homogeneous = np.dot(transform_matrix, xyz_camera_homogeneous.T).T

    # 去除齐次坐标
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

if __name__ == "__main__":
    save_root = "./log_dir/visu"
    gap_root = "./datasets/GAPartNet/dataset/data/"

    
    