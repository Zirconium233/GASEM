import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12357'
from network.gpv_layers import *
from datasets.datasets_pair import *
from network.sym_v1 import *
from loss.utils import *
from network.utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import argparse

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        # init_method="file:///home/zhangran/desktop/tmp/.shared_tmp",
        world_size=world_size
    )
    torch.manual_seed(0+rank)

def cleanup():
    dist.destroy_process_group()

def get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=False, shot=False):
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
        max_points=2000,
        augmentation=True,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
    )

    dataset_test_intra = GAPartNetPair(
        Path(test_intra_dir) / "pth",
        Path(test_intra_dir) / "meta",
        shuffle=False,
        max_points=2000,
        augmentation=True,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
    )

    dataset_test_inter = GAPartNetPair(
        Path(test_inter_dir) / "pth",
        Path(test_inter_dir) / "meta",
        shuffle=False,
        max_points=2000,
        augmentation=True,
        voxelization=voxelization, 
        group_size=2,
        voxel_size=[0.01,0.01,0.01],
        few_shot=few_shot,
        few_shot_num=few_shot_num,
        pos_jitter=0.1,
        with_pose=True,
        color_jitter=0.3,
        flip_prob=0.3,
        rotate_prob=0.3,
    )

    return dataset_train, dataset_test_intra, dataset_test_inter

def get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, batch_size=16, num_workers=8):
    train_sampler = DistributedSampler(dataset_train)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_utils.trivial_batch_collator,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler
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

# postnet9d
class FaceRecon_feat(nn.Module):
    def __init__(self, gcn_n_num, gcn_sup_num):
        super(FaceRecon_feat, self).__init__()
        self.neighbor_num = gcn_n_num
        self.support_num = gcn_sup_num

        # 3D convolution for point cloud
        self.conv_0 = Conv_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = Conv_layer(128, 128, support_num=self.support_num)
        self.pool_1 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_2 = Conv_layer(128, 256, support_num=self.support_num)
        self.conv_3 = Conv_layer(256, 256, support_num=self.support_num)
        self.pool_2 = Pool_layer(pooling_rate=4, neighbor_num=4)
        self.conv_4 = Conv_layer(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)", # type: ignore
                # cat_id: "tensor (bs, 1)",
                ):
        """
        Return: (bs, vertice_num, class_num)
        """

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = F.relu(self.conv_0(neighbor_index, vertices), inplace=True)

        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        neighbor_index = get_neighbor_index(v_pool_1,
                                                  min(self.neighbor_num, v_pool_1.shape[1] // 8))
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(1, 2)).transpose(1, 2),
                      inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        neighbor_index = get_neighbor_index(v_pool_2, min(self.neighbor_num,
                                                                v_pool_2.shape[1] // 8))
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)
        nearest_pool_1 = get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = get_nearest_index(vertices, v_pool_2)
        fm_2 = indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        '''
        feat_face = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4], dim=2)
        feat_face = torch.mean(feat_face, dim=1, keepdim=True)  # bs x 1 x channel
        feat_face_re = feat_face.repeat(1, feat.shape[1], 1)
        '''
        return feat

class PoseNet9D_Only_R(nn.Module):
    def __init__(self, feat_c_R=1280, R_c=4, gcn_n_num=10, gcn_sup_num=7, face_recon_c=6 * 5, obj_c=6, feat_face=768, feat_c_ts=1289, Ts_c=6):
        super(PoseNet9D_Only_R, self).__init__()
        self.rot_green = Rot_green(feat_c_R, R_c)
        self.rot_red = Rot_red(feat_c_R, R_c)
        self.face_recon = FaceRecon_feat(gcn_n_num, gcn_sup_num)
        # self.ts = Pose_Ts(feat_c_ts, Ts_c)

    def forward(self, points):
        bs, p_num = points.shape[0], points.shape[1]
        feat = self.face_recon(points - points.mean(dim=1, keepdim=True))
        #  rotation
        green_R_vec = self.rot_green(feat.permute(0, 2, 1))  # b x 4
        red_R_vec = self.rot_red(feat.permute(0, 2, 1))   # b x 4
        # normalization
        p_green_R = green_R_vec[:, 1:] / (torch.norm(green_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        p_red_R = red_R_vec[:, 1:] / (torch.norm(red_R_vec[:, 1:], dim=1, keepdim=True) + 1e-6)
        # sigmoid for confidence
        f_green_R = F.sigmoid(green_R_vec[:, 0])
        f_red_R = F.sigmoid(red_R_vec[:, 0])
        # translation and size no need
        return p_green_R, p_red_R, f_green_R, f_red_R
    
# loss
class fs_net_loss_R(nn.Module):
    def __init__(self, loss_type="smoothl1"):
        super(fs_net_loss_R, self).__init__()
        if loss_type == 'l1':
            self.loss_func_t = nn.L1Loss()
            self.loss_func_s = nn.L1Loss()
            self.loss_func_Rot1 = nn.L1Loss()
            self.loss_func_Rot2 = nn.L1Loss()
            self.loss_func_r_con = nn.L1Loss()
            self.loss_func_Recon = nn.L1Loss()
        elif loss_type == 'smoothl1':   # same as MSE
            self.loss_func_t = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_s = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot1 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Rot2 = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_r_con = nn.SmoothL1Loss(beta=0.5)
            self.loss_func_Recon = nn.SmoothL1Loss(beta=0.3)
        else:
            raise NotImplementedError

    def forward(self, pred_list, gt_list, sym):
        loss_list = {}

        self.rot_1_w = 1

        loss_list["Rot1"] = self.rot_1_w * self.cal_loss_Rot1(pred_list["Rot1"], gt_list["Rot1"])

        loss_list["Rot2"] = self.rot_1_w * self.cal_loss_Rot2(pred_list["Rot2"], gt_list["Rot2"], sym)

        # loss_list["Recon"] = self.recon_w * self.cal_loss_Recon(pred_list["Recon"], gt_list["Recon"])

        # loss_list["Tran"] = self.tran_w * self.cal_loss_Tran(pred_list["Tran"], gt_list["Tran"])
    
        # loss_list["Size"] = self.size_w * self.cal_loss_Size(pred_list["Size"], gt_list["Size"])

        return loss_list

    def cal_loss_Rot1(self, pred_v, gt_v):
        bs = pred_v.shape[0]
        res = torch.zeros([bs], dtype=torch.float32, device=pred_v.device)
        for i in range(bs):
            pred_v_now = pred_v[i, ...]
            gt_v_now = gt_v[i, ...]
            res[i] = self.loss_func_Rot1(pred_v_now, gt_v_now)
        res = torch.mean(res)
        return res

    def cal_loss_Rot2(self, pred_v, gt_v, sym):
        bs = pred_v.shape[0]
        res = 0.0
        valid = 0.0
        for i in range(bs):
            sym_now = sym[i, 0]
            if sym_now == 1:
                continue
            else:
                pred_v_now = pred_v[i, ...]
                gt_v_now = gt_v[i, ...]
                res += self.loss_func_Rot2(pred_v_now, gt_v_now)
                valid += 1.0
        if valid > 0.0:
            res = res / valid
        return res

    def cal_loss_Recon(self, pred_recon, gt_recon):
        return self.loss_func_Recon(pred_recon, gt_recon)

    def cal_loss_Tran(self, pred_trans, gt_trans):
        return self.loss_func_t(pred_trans, gt_trans)

    def cal_loss_Size(self, pred_size, gt_size):
        return self.loss_func_s(pred_size, gt_size)

class test_GPV(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = PoseNet9D_Only_R()
    
    def forward(self, pc_list: List[PointCloudPair]):
        points1 = torch.cat([pc.pc1.points.unsqueeze(0) for pc in pc_list], dim=0)  # pc_list is batch size
        points2 = torch.cat([pc.pc2.points.unsqueeze(0) for pc in pc_list], dim=0)
        p_green_R1, p_red_R1, f_green_R1, f_red_R1 = self.backbone(points1[:, :, 0:3])
        p_green_R2, p_red_R2, f_green_R2, f_red_R2 = self.backbone(points2[:, :, 0:3])
        return (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2)

# Helper function to extract ground truth rotation vectors from the batch of PointCloudPairs
def ground_truth_rotations(rot_list: List[torch.Tensor]) -> np.ndarray:
    rotations = []
    for rot in rot_list:
        # Assuming the rotations are stored as 3x3 matrices in pc_pair.rot_1 and pc_pair.rot_2
        rotation_matrix = np.array(rot.cpu())  # Example using rot_1, adjust as needed
        rotations.append(rotation_matrix)
    return torch.tensor(np.stack(rotations))

def train(rank, world_size, model_class, root_dir, test_intra_dir, test_inter_dir, lr, num_epochs, log_dir):
    setup(rank, world_size)
    # 创建并分配Sampler
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(
        root_dir, test_intra_dir, test_inter_dir, shot=False
    )
    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank)
    dataloader_train = DataLoader(dataset_train, batch_size=16, sampler=train_sampler, num_workers=0, pin_memory=True, collate_fn=data_utils.trivial_batch_collator,)
    test_intra_sampler = DistributedSampler(dataset_test_intra, num_replicas=world_size, rank=rank)
    dataloader_test_intra = DataLoader(dataset_test_intra, batch_size=16, sampler=test_intra_sampler, num_workers=0, pin_memory=True, collate_fn=data_utils.trivial_batch_collator,)
    test_inter_sampler = DistributedSampler(dataset_test_inter, num_replicas=world_size, rank=rank)
    dataloader_test_inter = DataLoader(dataset_test_inter, batch_size=16, sampler=test_inter_sampler, num_workers=0, pin_memory=True, collate_fn=data_utils.trivial_batch_collator,)

    model = model_class()
    model = model.to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    dist.barrier()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = fs_net_loss_R()
    log_dir = log_dir + "/" + str(datetime.today())
    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    global_step = 0
    print("_________________________train_epoch___________________________")
    for epoch in range(num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0
        if epoch == 0:
            # first test epoch
            print("______________________first_test_epoch_________________________")
            if rank == 0:
                torch.save(model.module.state_dict(), log_dir + r'/' + f"GPV_[{epoch+1}|{num_epochs}].pth")
            dist.barrier()
            test_metrics(rank, world_size, model, dataloader_test_inter, epoch, 'test_inter', writer)
            test_metrics(rank, world_size, model, dataloader_test_intra, epoch, 'test_intra', writer)
        # print(f"{rank} arrived barrier 1")
        dist.barrier()
        for batch_idx, batch in enumerate(tqdm(dataloader_train, leave=True)):
            pc_pairs = [pair.to(rank) for pair in batch]
            optimizer.zero_grad()

            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)

            # Assuming we have ground truth rotations
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2 for pc in pc_pairs]))  # Function to get ground truth rotation vectors

            pred_list1 = {
                "Rot1": p_green_R1,
                "Rot2": p_red_R1,
            }
            gt_list1 = {
                "Rot1": R_green_gt1.to(rank),
                "Rot2": R_red_gt1.to(rank),
            }

            pred_list2 = {
                "Rot1": p_green_R2,
                "Rot2": p_red_R2,
            }
            gt_list2 = {
                "Rot1": R_green_gt2.to(rank),
                "Rot2": R_red_gt2.to(rank),
            }

            sym1, sym2 = get_sym_from_input(pc_pairs)

            loss_dict1 = criterion(pred_list1, gt_list1, sym1)
            loss_dict2 = criterion(pred_list2, gt_list2, sym2)
            loss = (loss_dict1['Rot1'] + loss_dict1['Rot2'] + loss_dict2['Rot1'] + loss_dict2['Rot2']) / 2.0
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            # 每10个batch记录一次loss
            if (batch_idx + 1) % 10 == 0:
                loss_det = loss.detach()
                dist.all_reduce(loss_det, dist.ReduceOp.SUM)
                loss_det /= world_size
                if rank == 0:
                    writer.add_scalar('train/loss', loss_det.item(), global_step)
                    print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")

        avg_loss = total_loss / len(dataloader_train)
        t = torch.tensor(avg_loss).to(rank)
        dist.all_reduce(t)
        t /= world_size
        avg_loss = t.item()
        if rank == 0:
            print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
            writer.add_scalar('train/avg_loss', avg_loss, epoch)
        dist.barrier()

        # 每5个epoch跑一次测试集
        if (epoch + 1) % 5 == 0:
            if rank == 0:
                torch.save(model.module.state_dict(), log_dir + r'/' + f"GPV_[{epoch+1}|{num_epochs}].pth")
            dist.barrier()
            test_metrics(rank, world_size, model, dataloader_test_inter, epoch, 'test_inter', writer)
            test_metrics(rank, world_size, model, dataloader_test_intra, epoch, 'test_intra', writer)
        dist.barrier()
        # print(f"{rank} arrived barrier 2")
    cleanup()


def test_metrics(rank, world_size, model, dataloader , epoch, phase, writer):
    print("______________________" + phase + "_______________________")
    model.eval()
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    dataloader.sampler
    with torch.no_grad():
        for batch in tqdm(dataloader, leave=True):
            pc_pairs = [pair.to(rank) for pair in batch]
            # print(f"rank: {rank}")
            # print(model.device)
            # print(pc_pairs[0].rot_1.device)
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)

            # Assuming we have ground truth rotations
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
            # Convert predicted vectors and ground truth vectors back to rotation matrices
            pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1)
            pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2)
            gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1)
            gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2)

            # Store predictions and ground truths for metrics calculation
            all_pred_rot_matrices.append(pred_rot_matrices1.cpu())
            all_pred_rot_matrices.append(pred_rot_matrices2.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices1.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices2.cpu())

    all_pred_rot_matrices = torch.cat(all_pred_rot_matrices, dim=0)
    all_gt_rot_matrices = torch.cat(all_gt_rot_matrices, dim=0)

    mean_rot_error = calculate_pose_metrics(
        all_pred_rot_matrices, all_gt_rot_matrices
    )
    
    t = torch.tensor(mean_rot_error).to(rank)
    dist.all_reduce(t)
    t /= world_size
    mean_rot_error = t.item()
    if writer is not None:
        writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
        print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    model.train()


if __name__ == "__main__":
    root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
    test_intra_dir = "/16T/zhangran/GAPartNet_re_rendered/test_intra"
    test_inter_dir = "/16T/zhangran/GAPartNet_re_rendered/test_inter"
    
    # model = test_GPV()
    world_size = torch.cuda.device_count()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    # args = parser.parse_args()
    # train(args.local_rank, world_size, test_GPV, dataset_train, dataset_test_intra, dataset_test_inter, 0.001, 100, "./log_dir/GPV_test_sym_v1_muti")
    # mp.spawn(train, args=(world_size, test_GPV, dataset_train, dataset_test_intra, dataset_test_inter, 0.001, 100, "./log_dir/GPV_test_sym_v1_muti"), nprocs=world_size, join=True)
    # model.load_state_dict(torch.load('./log_dir/GPV_test/2024-05-22 01:41:18.391449/GPV_[40|40].pth'))
    # train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, 0.001, 40, "./log_dir/GPV_test_sym_v1")
    mp.spawn(train, args=(world_size, test_GPV, root_dir, test_intra_dir, test_inter_dir, 0.001, 100, "./log_dir/GPV_test_sym_v1_muti"), nprocs=world_size, join=True)
