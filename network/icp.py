import torch
import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import open3d.t.pipelines.registration as treg
import torch.nn.functional as F

# Iterative Closest Point, between pc1 :[20000,6] and pc2: [20000,6], 0:3 is position, 3:6 is color
def icp(p1: torch.Tensor, p2: torch.Tensor, cat_with_color=False, origin_point=False):
    def process_single_pair(pc1, pc2):
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc1.astype(np.float64)) # use float64 to lift efficiency, see https://github.com/isl-org/Open3D/issues/1045 for details
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc2.astype(np.float64))
        
        # use ICP just for alignment, you can choose weather to use the flows between original points and aligned points
        threshold = 0.02
        trans_init = np.eye(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        pcd2.transform(reg_p2p.transformation)
        transformed_pc2 = np.asarray(pcd2.points)
        
        # find the nearest neighbor
        pc1_tensor = torch.tensor(pc1, dtype=torch.float32).cuda()
        pc2_tensor = torch.tensor(transformed_pc2, dtype=torch.float32).cuda()
        pc2_original = torch.tensor(pc2, dtype=torch.float32).cuda()
        dist = torch.cdist(pc1_tensor, pc2_tensor)
        min_dist, indices = torch.min(dist, dim=1)
        if origin_point:
            matched_pc2 = pc2_original[indices]
        else:
            matched_pc2 = pc2_tensor[indices]
        flows = matched_pc2 - pc1_tensor
        
        return flows
    
    if p1.ndim == 3 and p2.ndim == 3:
        batch_size, num_points, _ = p1.shape
        flows_batch = torch.zeros_like(p1)
        
        for i in range(batch_size):
            pc1 = p1[i, :, :3].cpu().numpy()
            pc2 = p2[i, :, :3].cpu().numpy()
            flows = process_single_pair(pc1, pc2)
            flows_tensor = flows
            flows_batch[i, :, :3] = flows_tensor
            flows_batch[i, :, 3:6] = p1[i, :, 3:6]
        if cat_with_color:
            return flows_batch
        else:
            return flows_batch[:, :, :3]
    
    elif p1.ndim == 2 and p2.ndim == 2:
        pc1 = p1[:, :3].cpu().numpy()
        pc2 = p2[:, :3].cpu().numpy()
        flows = process_single_pair(pc1, pc2)
        flows_tensor = torch.from_numpy(flows).float().to(p1.device)
        if cat_with_color:
            flows_tensor = torch.cat((flows_tensor, p1[:, 3:6]), dim=1)
        
        return flows_tensor
    
    else:
        raise ValueError("Input tensors must be either 2D or 3D with batch dimension.")

 # same like icp, but only return indices, and p2[indice] is the matched points
def icp_mask(p1: torch.Tensor, p2: torch.Tensor, directly_nn=False):
    def process_single_pair(pc1, pc2, directly_nn=False):
        if not directly_nn:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pc1.astype(np.float64))
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pc2.astype(np.float64))
            threshold = 0.02
            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd1, pcd2, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            pcd2.transform(reg_p2p.transformation)
            transformed_pc2 = np.asarray(pcd2.points)
        else:
            transformed_pc2 = pc2
        
        pc1_tensor = torch.tensor(pc1, dtype=torch.float32).cuda()
        pc2_tensor = torch.tensor(transformed_pc2, dtype=torch.float32).cuda()
        dist = torch.cdist(pc1_tensor, pc2_tensor)
        _, indices = torch.min(dist, dim=1)
        
        return indices.cpu().numpy()
    
    if p1.ndim == 3 and p2.ndim == 3:
        if directly_nn:
            pc1_tensor = p1[:, :, :3].float().cuda()
            pc2_tensor = p2[:, :, :3].float().cuda()
            batch_size = pc1_tensor.size(0)
            
            indices_batch = torch.zeros(batch_size, p1.size(1), dtype=torch.long)
            
            for i in range(batch_size):
                dist = torch.cdist(pc1_tensor[i], pc2_tensor[i])
                _, indices = torch.min(dist, dim=1)
                indices_batch[i] = indices.cpu()
            
            return indices_batch
        else:
            batch_size, num_points, _ = p1.shape
            indices_batch = torch.zeros(batch_size, num_points, dtype=torch.long)
            
            for i in range(batch_size):
                pc1 = p1[i, :, :3].cpu().numpy()
                pc2 = p2[i, :, :3].cpu().numpy()
                indices = process_single_pair(pc1, pc2, directly_nn)
                indices_batch[i, :] = torch.tensor(indices, dtype=torch.long)
            
            return indices_batch
    
    elif p1.ndim == 2 and p2.ndim == 2:
        pc1 = p1[:, :3].cpu().numpy()
        pc2 = p2[:, :3].cpu().numpy()
        indices = process_single_pair(pc1, pc2, directly_nn)
        indices_tensor = torch.from_numpy(indices).long().to(p1.device)
        
        return indices_tensor
    
    else:
        raise ValueError("Input tensors must be either 2D or 3D with batch dimension.")

def icp_gpu(p1: torch.Tensor, p2: torch.Tensor, cat_with_color=False, origin_point=False, voxel_sizes=[0.05, 0.025, 0.0125], max_correspondence_distances=[0.14, 0.07, 0.03], criteria_list=None):
    if criteria_list is None:
        criteria_list = [
            treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                        relative_rmse=0.0001,
                                        max_iteration=20),
            treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
            treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
        ]
    voxel_sizes_vector = o3d.utility.DoubleVector(voxel_sizes)
    max_corr_dist_vector = o3d.utility.DoubleVector(max_correspondence_distances)
    
    def process_single_pair(pc1, pc2):
        pcd1 = o3d.t.geometry.PointCloud(o3d.core.Tensor(pc1, o3d.core.Dtype.Float32))
        pcd2 = o3d.t.geometry.PointCloud(o3d.core.Tensor(pc2, o3d.core.Dtype.Float32))

        pcd1_cuda = pcd1.cuda(0)
        pcd2_cuda = pcd2.cuda(0)

        trans_init = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)

        # Multi-Scale ICP
        reg_ms_icp = treg.multi_scale_icp(
            pcd1_cuda, pcd2_cuda,
            voxel_sizes_vector, criteria_list,
            max_corr_dist_vector,
            trans_init,
            treg.TransformationEstimationPointToPoint(),
            # callback_after_iteration=callback_after_iteration
        )

        pcd2_cuda.transform(reg_ms_icp.transformation)
        transformed_pc2 = np.asarray(pcd2_cuda.point.positions.cpu().numpy(), dtype=np.float32)

        pc1_tensor = torch.tensor(pc1, dtype=torch.float32).cuda()
        pc2_tensor = torch.tensor(transformed_pc2, dtype=torch.float32).cuda()
        pc2_original = torch.tensor(pc2, dtype=torch.float32).cuda()
        dist = torch.cdist(pc1_tensor, pc2_tensor)
        min_dist, indices = torch.min(dist, dim=1)
        if origin_point:
            matched_pc2 = pc2_original[indices]
        else:
            matched_pc2 = pc2_tensor[indices]

        flows = matched_pc2 - pc1_tensor

        return flows
    
    if p1.ndim == 3 and p2.ndim == 3:
        batch_size, num_points, _ = p1.shape
        flows_batch = torch.zeros_like(p1)

        for i in range(batch_size):
            pc1 = p1[i, :, :3].cpu().numpy()
            pc2 = p2[i, :, :3].cpu().numpy()
            flows = process_single_pair(pc1, pc2)
            flows_tensor = flows.to(p1.device)
            flows_batch[i, :, :3] = flows_tensor
            flows_batch[i, :, 3:6] = p1[i, :, 3:6]
        if cat_with_color:
            return flows_batch
        else:
            return flows_batch[:, :, :3]
    
    elif p1.ndim == 2 and p2.ndim == 2:
        pc1 = p1[:, :3].cpu().numpy()
        pc2 = p2[:, :3].cpu().numpy()
        flows = process_single_pair(pc1, pc2)
        flows_tensor = torch.from_numpy(flows).float().to(p1.device)
        if cat_with_color:
            flows_tensor = torch.cat((flows_tensor, p1[:, 3:6]), dim=1)
        
        return flows_tensor
    
    else:
        raise ValueError("Input tensors must be either 2D or 3D with batch dimension.")

def icp_gpu_torch(p1: torch.Tensor, p2: torch.Tensor, cat_with_color=False, max_iter=20, tol=1e-6):
    def nearest_neighbor(src, dst):
        dist = torch.cdist(src, dst)
        indices = torch.argmin(dist, dim=1)
        return indices
    
    def compute_transformation(src, dst):
        src_mean = torch.mean(src, dim=0)
        dst_mean = torch.mean(dst, dim=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.t() @ dst_centered
        U, S, Vt = torch.svd(H)
        R = Vt.t() @ U.t()
        t = dst_mean - R @ src_mean
        return R, t
    
    def process_single_pair(pc1, pc2):
        src = pc1.clone()
        dst = pc2.clone()

        for _ in range(max_iter):
            indices = nearest_neighbor(src, dst)
            src_matched = src
            dst_matched = dst[indices]

            R, t = compute_transformation(src_matched, dst_matched)

            src = src @ R.t() + t

            error = F.mse_loss(src, pc1)
            if error < tol:
                break
        
        flows = src - pc1
        return flows
    
    if p1.ndim == 3 and p2.ndim == 3:
        batch_size, num_points, _ = p1.shape
        flows_batch = torch.zeros_like(p1)

        for i in range(batch_size):
            pc1 = p1[i, :, :3].to(p1.device)
            pc2 = p2[i, :, :3].to(p1.device)
            flows = process_single_pair(pc1, pc2)
            flows_batch[i, :, :3] = flows
            if cat_with_color:
                flows_batch[i, :, 3:6] = p1[i, :, 3:6]
        if cat_with_color:
            return flows_batch
        else:
            return flows_batch[:, :, :3]
    
    elif p1.ndim == 2 and p2.ndim == 2:
        pc1 = p1[:, :3].to(p1.device)
        pc2 = p2[:, :3].to(p1.device)
        flows = process_single_pair(pc1, pc2)
        if cat_with_color:
            flows = torch.cat((flows, p1[:, 3:6]), dim=1)
        
        return flows
    
    else:
        raise ValueError("Input tensors must be either 2D or 3D with batch dimension.")