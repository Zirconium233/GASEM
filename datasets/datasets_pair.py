import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Tuple, Union, List
from glob import glob
from GAPartNet.dataset.point_cloud import PointCloud, PointCloudBatch
from GAPartNet.dataset import data_utils
# from GAPartNet.dataset.gapartnet import *
from epic_ops.voxelize import voxelize
from GAPartNet.misc.info import OBJECT_NAME2ID
import random
import copy
import tqdm
from pointnet2_ops import pointnet2_utils as futils

class GAPartNetPair(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path] = "",
        meta_dir: Union[str, Path] = "",
        idx_dir: Union[str, Path] = None,
        shuffle: bool = False,
        max_points: int = 20000,
        group_size: int = 2,
        augmentation: bool = False,
        voxelization: bool = False,
        voxel_size: Tuple[float, float, float] = (1 / 100, 1 / 100, 1 / 100),
        few_shot = False,
        few_shot_num = 512,
        pos_jitter: float = 0.,
        color_jitter: float = 0.,
        flip_prob: float = 0.,
        rotate_prob: float = 0.,
        nopart_path: str = "./datasets/GAPartNet/data/nopart.txt",
        no_label = False,
        with_pose = True,
        glob_condition: str = "/*.pth", 
        choose_category: List = None,
        no_ball_space: bool = False,
    ):
        super().__init__()
        file_paths=glob(str(root_dir) + glob_condition)
        if os.path.exists(nopart_path):
            self.nopart_files = open(nopart_path, "r").readlines()[0].split(" ")
            self.nopart_names = [p.split("/")[-1].split(".")[0] for p in self.nopart_files]
            file_paths = [path for path in file_paths 
                        if path.split("/")[-1].split(".")[0] not in self.nopart_names]
        file_paths.sort()
        if choose_category is not None:
            file_paths = [path for path in file_paths if path.split("/")[-1].split(".")[0].split('_')[0] in choose_category]
        self.pc_paths = file_paths
        self.no_label = no_label
        self.augmentation = augmentation
        self.voxelization = voxelization
        self.pos_jitter = pos_jitter
        self.color_jitter = color_jitter
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.with_pose = with_pose
        self.voxel_size = voxel_size
        self.max_points = max_points
        self.group_size = group_size
        self.no_ball_space = no_ball_space
        self.meta_dir = str(meta_dir)
        self.idx_dir = str(idx_dir)
        if self.group_size != 2:
            raise NotImplementedError
        self.group_files = []
        for i in range(len(file_paths)-1):
            name = file_paths[i].split('/')[-1].split('.')[0]
            splited_name = name.split('_')
            assert len(splited_name) == 4, f"name format error, the correct format is ObjName_ObjId_SequenceId_Index, but got {name} "
            category, obj_id, sequence_id, id = splited_name
            name_ = file_paths[i+1].split('/')[-1].split('.')[0]
            splited_name_ = name_.split('_')
            assert len(splited_name_) == 4, f"name format error, the correct format is ObjName_ObjId_SequenceId_Index, but got {name_} "
            category_, obj_id_, sequence_id_, id_ = splited_name_
            if category == category_ and obj_id == obj_id_ and sequence_id == sequence_id_:
                self.group_files.append((file_paths[i], file_paths[i+1]))
            else:
                continue
        if shuffle:
            random.shuffle(self.group_files)
        if few_shot:
            self.group_files = self.group_files[:few_shot_num]

    def __len__(self):
        return len(self.group_files)
    
    def __getitem__(self, idx) -> "PointCloudPair":
        path1, path2 = self.group_files[idx]
        name1 = path1.split('/')[-1].split('.')[0]
        name2 = path2.split('/')[-1].split('.')[0]
        meta1 = self.meta_dir + f"/{name1}.json"
        meta2 = self.meta_dir + f"/{name2}.json"
        if self.with_pose:
            file1, rotate1, trans1, t_cw_1 = load_data(path1, no_label = self.no_label, meta_path=meta1, with_pose=self.with_pose)
            file2, rotate2, trans2, t_cw_2 = load_data(path2, no_label = self.no_label, meta_path=meta2, with_pose=self.with_pose)
            rotate1 = torch.tensor(rotate1, dtype=torch.float32)
            rotate2 = torch.tensor(rotate2, dtype=torch.float32)
        else:
            file1 = load_data(path1, no_label = self.no_label, meta_path=meta1, with_pose=self.with_pose)
            file2 = load_data(path2, no_label = self.no_label, meta_path=meta2, with_pose=self.with_pose)
            # import pdb; pdb.set_trace() # nopart
        # make instance label continuous
        if self.idx_dir != 'None':
            idx1_path = self.idx_dir + "/" + name1 + "_idx.pth"
            idx2_path = self.idx_dir + "/" + name2 + "_idx.pth"
        else:
            idx1_path = None
            idx2_path = None
        file1 = downsample(file1, max_points=self.max_points, fps_idx=idx1_path)
        file2 = downsample(file2, max_points=self.max_points, fps_idx=idx2_path)
        if not bool((file1.instance_labels != -100).any()) or not bool((file2.instance_labels != -100).any()):
            return self.__getitem__((idx+1) % self.__len__())
        file1 = compact_instance_labels(file1)
        file2 = compact_instance_labels(file2)
        # apply augmentations
        if self.augmentation:
            file1 = apply_augmentations(file1, 
                pos_jitter=self.pos_jitter,
                color_jitter=self.color_jitter,
                flip_prob=self.flip_prob,
                rotate_prob=self.rotate_prob,)
            file2 = apply_augmentations(file2, 
                pos_jitter=self.pos_jitter,
                color_jitter=self.color_jitter,
                flip_prob=self.flip_prob,
                rotate_prob=self.rotate_prob,)
        # add information about instance. 
        file1 = generate_inst_info(file1)
        file2 = generate_inst_info(file2)
        if self.voxelization:
            file1 = file1.to_tensor()
            file2 = file2.to_tensor()
            file1 = apply_voxelization(file1, voxel_size=self.voxel_size)
            file2 = apply_voxelization(file2, voxel_size=self.voxel_size)
        if self.with_pose and not self.no_ball_space:
            file1 = file1.to_tensor()
            file2 = file2.to_tensor()
            return PointCloudPair(file1, file2, rotate1, rotate2)
        elif self.with_pose and self.no_ball_space:
            s1 = trans1[0]
            s2 = trans2[0]
            t1 = trans1[1:4]
            t2 = trans2[1:4]
            file1.points[:,0:3] = file1.points[:,0:3] * s1 + t1
            file2.points[:,0:3] = file2.points[:,0:3] * s2 + t2
            t1 = torch.tensor(t_cw_1, dtype=torch.float32)
            t2 = torch.tensor(t_cw_2, dtype=torch.float32)
            s1 = torch.tensor(s1, dtype=torch.float32)
            s2 = torch.tensor(s2, dtype=torch.float32)
            file1 = file1.to_tensor()
            file2 = file2.to_tensor()
            return PointCloudPair(file1, file2, rotate1, rotate2, t1, t2, s1, s2)
        else:
            file1 = file1.to_tensor()
            file2 = file2.to_tensor()
            return PointCloudPair(file1, file2)

def apply_augmentations(
    pc: PointCloud,
    *,
    pos_jitter: float = 0.,
    color_jitter: float = 0.,
    flip_prob: float = 0.,
    rotate_prob: float = 0.,
) -> PointCloud:
    pc = copy.copy(pc)

    m = np.eye(3)
    if pos_jitter > 0:
        m += np.random.randn(3, 3) * pos_jitter

    if flip_prob > 0:
        if np.random.rand() < flip_prob:
            m[0, 0] = -m[0, 0]

    if rotate_prob > 0:
        if np.random.rand() < flip_prob:
            theta = np.random.rand() * np.pi * 2
            m = m @ np.asarray([
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ])

    pc.points = pc.points.copy()
    pc.points[:, :3] = pc.points[:, :3] @ m

    if color_jitter > 0:
        pc.points[:, 3:] += np.random.randn(
            1, pc.points.shape[1] - 3
        ) * color_jitter

    return pc


def downsample(pc: PointCloud, *, max_points: int = 20000, fps_idx=None) -> PointCloud:
    pc = copy.copy(pc)  # copy
    num_points = pc.points.shape[0]

    if num_points > max_points:
        # random choose max_points 
        if fps_idx is None:
            # indices = np.random.choice(num_points, max_points, replace=False)
            indices = range(max_points) # because the points is already sampled by fps. 
        else:
            indices = torch.load(fps_idx)
        # indices = farthest_point_sampling_cuda(pc.points, max_points)
        pc.points = pc.points[indices]
        if pc.sem_labels is not None:
            pc.sem_labels = pc.sem_labels[indices]
        if pc.instance_labels is not None:
            pc.instance_labels = pc.instance_labels[indices]
        if pc.gt_npcs is not None:
            pc.gt_npcs = pc.gt_npcs[indices]

    return pc


def farthest_point_sampling(points, num_samples):
    """
    fps 
    :param points:  (N, D)
    :param num_samples: nums of target points
    :return: indices of target points
    """
    N, D = points.shape
    centroids = np.zeros(num_samples, dtype=np.int64)
    distances = np.ones(N) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(num_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    
    return centroids

def farthest_point_sampling_cuda(points, num_samples):
    idx = futils.furthest_point_sample(torch.tensor(points).unsqueeze(0).cuda(), num_samples).long()
    return idx.cpu().numpy()[0]

def compact_instance_labels(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    valid_mask = pc.instance_labels >= 0
    instance_labels = pc.instance_labels[valid_mask]
    _, instance_labels = np.unique(instance_labels, return_inverse=True)
    pc.instance_labels[valid_mask] = instance_labels

    return pc


def generate_inst_info(pc: PointCloud) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]

    num_instances = int(pc.instance_labels.max()) + 1
    instance_regions = np.zeros((num_points, 9), dtype=np.float32)
    num_points_per_instance = []
    instance_sem_labels = []
    
    assert num_instances > 0

    for i in range(num_instances):
        indices = np.where(pc.instance_labels == i)[0]

        xyz_i = pc.points[indices, :3]
        min_i = xyz_i.min(0)
        max_i = xyz_i.max(0)
        mean_i = xyz_i.mean(0)
        instance_regions[indices, 0:3] = mean_i
        instance_regions[indices, 3:6] = min_i
        instance_regions[indices, 6:9] = max_i

        num_points_per_instance.append(indices.shape[0])
        instance_sem_labels.append(int(pc.sem_labels[indices[0]]))

    pc.num_instances = num_instances
    pc.instance_regions = instance_regions
    pc.num_points_per_instance = np.asarray(num_points_per_instance, dtype=np.int32)
    pc.instance_sem_labels = np.asarray(instance_sem_labels, dtype=np.int32)

    return pc

def load_data(file_path: str, no_label: bool = False, with_pose : bool = True, meta_path : str = None) -> Tuple[PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    if not no_label:
        pc_data = torch.load(file_path)
    else:
        # testing data type, e.g. real world point cloud without GT semantic label.
        raise NotImplementedError

    pc_id = file_path.split("/")[-1].split(".")[0]
    object_cat = OBJECT_NAME2ID[pc_id.split("_")[0]]

    if not with_pose:
        return PointCloud(
            pc_id=pc_id,
            obj_cat=object_cat,
            points=np.concatenate(
                [pc_data[0], pc_data[1]],
                axis=-1, dtype=np.float32,
            ),
            sem_labels=pc_data[2].astype(np.int64),
            instance_labels=pc_data[3].astype(np.int32),
            gt_npcs=pc_data[4].astype(np.float32),
        )
    else:
        assert meta_path is not None, "can't find meta file, the meta path is none"
        meta = None
        with open(meta_path, "r+", encoding='utf-8') as f:
            meta = json.loads(f.read())
        world2camera_rotation = np.array(meta['world2camera_rotation']).astype(np.float32).reshape(3, 3)
        camera2world_translation = np.array(meta['camera2world_translation']).astype(np.float32)
        trans = np.array(meta['scale_param']).astype(np.float32)
        return PointCloud(
            pc_id=pc_id,
            obj_cat=object_cat,
            points=np.concatenate(
                [pc_data[0], pc_data[1]],
                axis=-1, dtype=np.float32,
            ),
            sem_labels=pc_data[2].astype(np.int64),
            instance_labels=pc_data[3].astype(np.int32),
            gt_npcs=pc_data[4].astype(np.float32),
        ), np.array(world2camera_rotation, dtype=np.float32), np.array(trans, dtype=np.float32), np.array(camera2world_translation, dtype=np.float32)

def apply_voxelization(
    pc: PointCloud, *, voxel_size: Tuple[float, float, float]
) -> PointCloud:
    pc = copy.copy(pc)

    num_points = pc.points.shape[0]
    pt_xyz = pc.points[:, :3]
    points_range_min = pt_xyz.min(0)[0] - 1e-4
    points_range_max = pt_xyz.max(0)[0] + 1e-4
    voxel_features, voxel_coords, _, pc_voxel_id = voxelize(
        pt_xyz, pc.points,
        batch_offsets=torch.as_tensor([0, num_points], dtype=torch.int64, device = pt_xyz.device),
        voxel_size=torch.as_tensor(voxel_size, device = pt_xyz.device),
        points_range_min=torch.as_tensor(points_range_min, device = pt_xyz.device),
        points_range_max=torch.as_tensor(points_range_max, device = pt_xyz.device),
        reduction="mean",
    )
    assert (pc_voxel_id >= 0).all()

    voxel_coords_range = (voxel_coords.max(0)[0] + 1).clamp(min=128, max=None)

    pc.voxel_features = voxel_features
    pc.voxel_coords = voxel_coords
    pc.voxel_coords_range = voxel_coords_range.tolist()
    pc.pc_voxel_id = pc_voxel_id

    return pc

class PointCloudPair:

    pc1 : PointCloud
    pc2 : PointCloud
    rot_1: Union[torch.Tensor, np.ndarray] = None
    rot_2: Union[torch.Tensor, np.ndarray] = None
    t1: Union[torch.Tensor, np.ndarray] = None
    t2: Union[torch.Tensor, np.ndarray] = None
    s1: Union[torch.Tensor, np.ndarray] = None
    s2: Union[torch.Tensor, np.ndarray] = None
    def __init__(self, pc_1=None, pc_2=None, rot_1=None, rot_2=None, t1=None, t2=None, s1=None, s2=None):
        if pc_1 != None and pc_2 != None:
            self.pc1 = copy.copy(pc_1)
            self.pc2 = copy.copy(pc_2)
            if rot_1 is not None and rot_2 is not None:
                self.rot_1 = copy.copy(rot_1)
                self.rot_2 = copy.copy(rot_2)
            if t1 is not None and t2 is not None:
                self.t1 = copy.copy(t1)
                self.t2 = copy.copy(t2)
            if s1 is not None and s2 is not None:
                self.s1 = copy.copy(s1)
                self.s2 = copy.copy(s2)

    def __getitem__(self, idx) -> PointCloud:
        assert idx == 0 or idx == 1, "idx out of range"
        if idx == 0:
            return self.pc1
        else:
            return self.pc2
    def to_tensor(self) -> "PointCloudPair":
        assert self.pc1 is not None and self.pc2 is not None, "pc1 or pc2 is None"
        pc1 = self.pc1.to_tensor()
        pc2 = self.pc2.to_tensor()

        if self.rot_1 is not None:
            assert self.rot_2 is not None, "one of rot is none"
            rot_1 = torch.tensor(self.rot_1)
            rot_2 = torch.tensor(self.rot_2)
        else:
            assert self.rot_2 is None, "one of rot is none"
            rot_1 = None
            rot_2 = None
        if self.t1 is not None:
            assert self.t2 is not None, "one of t is none"
            t1 = torch.tensor(self.t1)
            t2 = torch.tensor(self.t2)
        else:
            assert self.t2 is None, "one of t is none"
            t1 = None
            t2 = None
        if self.s1 is not None:
            assert self.s2 is not None, "one of s is none"
            s1 = torch.tensor(self.s1)
            s2 = torch.tensor(self.s2)
        else:
            assert self.s2 is None, "one of s is none"
            s1 = None
            s2 = None
        return PointCloudPair(pc1, pc2, rot_1, rot_2, t1, t2, s1, s2)
    
    def to(self, device: torch.device) -> "PointCloudPair":
        assert self.pc1 is not None and self.pc2 is not None, "pc1 or pc2 is None"
        pc1 = self.pc1.to(device)
        pc2 = self.pc2.to(device)
        if self.rot_1 is not None:
            assert self.rot_2 is not None, "one of rot is none"
            rot_1 = self.rot_1.to(device)
            rot_2 = self.rot_2.to(device)
        else:
            assert self.rot_2 is None, "one of rot is none"
            rot_1 = None
            rot_2 = None
        if self.t1 is not None:
            assert self.t2 is not None, "one of t is none"
            t1 = self.t1.to(device)
            t2 = self.t2.to(device)
        else:
            assert self.t2 is None, "one of t is none"
            t1 = None
            t2 = None
        if self.s1 is not None:
            assert self.s2 is not None, "one of s is none"
            s1 = self.s1.to(device)
            s2 = self.s2.to(device)
        else:
            assert self.s2 is None, "one of s is none"
            s1 = None
            s2 = None
        return PointCloudPair(pc1, pc2, rot_1, rot_2, t1, t2, s1, s2)
        
def get_datasets(root_dir, test_intra_dir, test_inter_dir, max_points=2000, voxelization=False, shot=False, choose_category: List=None, augmentation=True, with_pose=True, no_ball_space=False):
    if shot:
        few_shot = True
        few_shot_num = 20
    else:
        few_shot = False
        few_shot_num = None

    dataset_train = GAPartNetPair(
        Path(root_dir) / "pth",
        Path(root_dir) / "meta",
        shuffle=True,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=with_pose,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
        no_ball_space=no_ball_space,
    )

    dataset_test_intra = GAPartNetPair(
        Path(test_intra_dir) / "pth",
        Path(test_intra_dir) / "meta",
        shuffle=False,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=with_pose,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
        no_ball_space=no_ball_space,
    )

    dataset_test_inter = GAPartNetPair(
        Path(test_inter_dir) / "pth",
        Path(test_inter_dir) / "meta",
        shuffle=False,
        max_points=max_points,
        augmentation=augmentation,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=with_pose,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
        choose_category=choose_category,
        no_ball_space=no_ball_space,
    )

    return dataset_train, dataset_test_intra, dataset_test_inter

def get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, batch_size=16, num_workers=0):
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False
    )
    # test_intra_sampler = DistributedSampler(dataset_test_intra, shuffle=False)
    dataloader_test_intra = DataLoader(
        dataset_test_intra,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False,
        # sampler=test_intra_sampler
    )
    # test_inter_sampler = DistributedSampler(dataset_test_inter, shuffle=False)
    dataloader_test_inter = DataLoader(
        dataset_test_inter,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False,
        # sampler=test_inter_sampler
    )
    return dataloader_train, dataloader_test_intra, dataloader_test_inter


if __name__ == "__main__":
    root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
    dataset_train = GAPartNetPair(
                        Path(root_dir)  / "pth",
                        Path(root_dir)  / "meta",
                        Path(root_dir) / "idx",
                        shuffle=True,
                        max_points=2000,
                        augmentation=True,
                        voxelization=False, 
                        group_size=2,
                        voxel_size=[0.01,0.01,0.01],
                        few_shot=False,
                        few_shot_num=None,
                        pos_jitter=0.1,
                        with_pose=True,
                        color_jitter=0.3,
                        flip_prob=0.3,
                        rotate_prob=0.3,
                        no_ball_space=True,
                    )
    print(dataset_train[0].pc1.points.shape[0])
    print(dataset_train[0].rot_1.shape)
    print(dataset_train[0].t1.shape)
    print(dataset_train[0].s1.shape)
    print(dataset_train[0].s1)
    dataloader_train = DataLoader(
                        dataset_train,
                        batch_size=16,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=data_utils.trivial_batch_collator,
                        pin_memory=True,
                        drop_last=False,
                    )
    for i in tqdm.tqdm(dataset_train):
        pass