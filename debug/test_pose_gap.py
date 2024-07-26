import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.optim as optim
from network.gap_layers import *
from datasets.datasets_pair import *
import functools
from network.sym_v1 import *
from loss.utils import *
from network.utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime

class test_Sparse_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sparseunet = SparseUNet.build(6, [16, 32, 48, 64, 80, 96, 112], 2, functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1))
        self.rot_green_head = nn.Linear(16, 3)
        self.rot_red_head = nn.Linear(16, 3)
        

    def forward(self, pc_pairs: List[PointCloudPair]):
        pc1s = [pc_pair.pc1 for pc_pair in pc_pairs]
        pc2s = [pc_pair.pc2 for pc_pair in pc_pairs]
        bs = len(pc_pairs)
        pc_batch_1: PointCloudBatch = PointCloud.collate(pc1s)
        pc_batch_2: PointCloudBatch = PointCloud.collate(pc2s)
        
        voxel_tensor_1 = pc_batch_1.voxel_tensor
        pc_voxel_id_1 = pc_batch_1.pc_voxel_id
        voxel_features = self.sparseunet(voxel_tensor_1)
        pc_feature_1 = voxel_features.features[pc_voxel_id_1]

        voxel_tensor_2 = pc_batch_2.voxel_tensor
        pc_voxel_id_2 = pc_batch_2.pc_voxel_id
        voxel_features = self.sparseunet(voxel_tensor_2)
        pc_feature_2 = voxel_features.features[pc_voxel_id_2]

        pc_feature_1 = pc_feature_1.view(bs, -1, 16) # bs,n,16
        pc_feature_2 = pc_feature_2.view(bs, -1, 16)

        rot_green_1 = self.rot_green_head(pc_feature_1).mean(dim=1).view(bs, 3) # bs,3
        rot_green_2 = self.rot_green_head(pc_feature_2).mean(dim=1).view(bs, 3)

        rot_red_1 = self.rot_red_head(pc_feature_1).mean(dim=1).view(bs, 3)
        rot_red_2 = self.rot_red_head(pc_feature_2).mean(dim=1).view(bs, 3) # bs,3
        
        return (rot_green_1, rot_green_2), (rot_red_1, rot_red_2)
    
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
    

# Helper function to extract ground truth rotation vectors from the batch of PointCloudPairs
def ground_truth_rotations(rot_list: List[torch.Tensor]) -> np.ndarray:
    rotations = []
    for rot in rot_list:
        # Assuming the rotations are stored as 3x3 matrices in pc_pair.rot_1 and pc_pair.rot_2
        rotation_matrix = np.array(rot.cpu())  # Example using rot_1, adjust as needed
        rotations.append(rotation_matrix)
    return torch.tensor(np.stack(rotations))

def train(model: nn.Module, 
          dataloader_train: DataLoader, 
          dataloader_test_inter: DataLoader, 
          dataloader_test_intra: DataLoader, 
          lr: int = 0.001, 
          num_epochs: int=100, 
          log_dir: str=None, 
          device: torch.device=None):
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = fs_net_loss_R()
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    assert log_dir is not None, "No Log Dir"
    log_dir = log_dir + "/" + str(datetime.today())
    writer = SummaryWriter(log_dir=log_dir)

    global_step = 0
    print("_________________________train_epoch___________________________")
    for epoch in range(num_epochs):
        total_loss = 0
        if epoch == 0:
            # first test epoch
            print("______________________first_test_epoch_________________________")
            torch.save(model.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_metrics(model, dataloader_test_inter, device, writer, epoch, 'test_inter')
            test_metrics(model, dataloader_test_intra, device, writer, epoch, 'test_intra')
        for batch_idx, batch in enumerate(dataloader_train):
            pc_pairs = [pair.to(device) for pair in batch]
            optimizer.zero_grad()

            (p_green_R1, p_red_R1), (p_green_R2, p_red_R2) = model(pc_pairs)
            
            # Assuming we have ground truth rotations
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2 for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
            pred_list1 = {
                "Rot1": p_green_R1,
                "Rot2": p_red_R1,
            }
            gt_list1 = {
                "Rot1": R_green_gt1.cuda(),
                "Rot2": R_red_gt1.cuda(),
            }
            
            pred_list2 = {
                "Rot1": p_green_R2,
                "Rot2": p_red_R2,
            }
            gt_list2 = {
                "Rot1": R_green_gt2.cuda(),
                "Rot2": R_red_gt2.cuda(),
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
                writer.add_scalar('train/loss', loss.item(), global_step)
                print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)

        # 每5个epoch跑一次测试集
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_metrics(model, dataloader_test_inter, device, writer, epoch, 'test_inter')
            test_metrics(model, dataloader_test_intra, device, writer, epoch, 'test_intra')


def test_metrics(model, dataloader, device, writer, epoch, phase):
    print("______________________" + phase + "_______________________")
    model.eval()
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pc_pairs = [pair.to(device) for pair in batch]
            (p_green_R1, p_red_R1), (p_green_R2, p_red_R2) = model(pc_pairs)
            
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
    writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
    print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    model.train()

if __name__ == "__main__":
    root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
    test_intra_dir = "/16T/zhangran/GAPartNet_re_rendered/test_intra"
    test_inter_dir = "/16T/zhangran/GAPartNet_re_rendered/test_inter"
    dataset_train = GAPartNetPair(
                        Path(root_dir)  / "pth",
                        Path(root_dir)  / "meta",
                        shuffle=True,
                        max_points=20000,
                        augmentation=True,
                        voxelization=True, 
                        group_size=2,
                        voxel_size=[0.01,0.01,0.01],
                        few_shot=False,
                        few_shot_num=None,
                        pos_jitter=0.1,
                        with_pose=True,
                        color_jitter=0.3,
                        flip_prob=0.3,
                        rotate_prob=0.3,
                    )
    dataloader_train = DataLoader(
                        dataset_train,
                        batch_size=16,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=data_utils.trivial_batch_collator,
                        pin_memory=True,
                        drop_last=False,
                    )
    dataset_test_intra = GAPartNetPair(
                        Path(test_intra_dir)  / "pth",
                        Path(test_intra_dir)  / "meta",
                        shuffle=False,
                        max_points=20000,
                        augmentation=True,
                        voxelization=True, 
                        group_size=2,
                        voxel_size=[0.01,0.01,0.01],
                        few_shot=False,
                        few_shot_num=None,
                        # few_shot = True,
                        # few_shot_num = 20,
                        pos_jitter=0.1,
                        with_pose=True,
                        color_jitter=0.3,
                        flip_prob=0.3,
                        rotate_prob=0.3,
                    )
    dataloader_test_intra = DataLoader(
                        dataset_test_intra,
                        batch_size=16,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=data_utils.trivial_batch_collator,
                        pin_memory=True,
                        drop_last=False,
                    )
    dataset_test_inter = GAPartNetPair(
                        Path(test_inter_dir)  / "pth",
                        Path(test_inter_dir)  / "meta",
                        shuffle=False,
                        max_points=20000,
                        augmentation=True,
                        voxelization=True, 
                        group_size=2,
                        voxel_size=[0.01,0.01,0.01],
                        few_shot=False,
                        few_shot_num=None,
                        # few_shot = True,
                        # few_shot_num = 20,
                        pos_jitter=0.1,
                        with_pose=True,
                        color_jitter=0.3,
                        flip_prob=0.3,
                        rotate_prob=0.3,
                    )
    dataloader_test_inter = DataLoader(
                        dataset_test_inter,
                        batch_size=16,
                        shuffle=False,
                        num_workers=8,
                        collate_fn=data_utils.trivial_batch_collator,
                        pin_memory=True,
                        drop_last=False,
                    )
    model = test_Sparse_UNet()
    train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, 0.001, 40, "./log_dir/SparseUNet_test_sym_v1")
    torch.save(model.state_dict(), "/home/zhangran/desktop/tmp/SPNet_pose_backup.pth")