import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from network.gpv_layers import *
from datasets.datasets_pair import *
from network.sym_v1 import *
from loss.utils import *
from network.utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
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
        # rotation
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

root_dir = "/16T/zhangran/GAPartNet_re_rendered/train"
test_intra_dir = "/16T/zhangran/GAPartNet_re_rendered/test_intra"
test_inter_dir = "/16T/zhangran/GAPartNet_re_rendered/test_inter"
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


class test_GPV_Delta(nn.Module):
    def __init__(self):
        super().__init__()
        # self.backbone = PoseNet9D_Only_R()
        self.rot_green = Rot_green(1280, 4)
        self.rot_red = Rot_red(1280, 4)
        self.face_recon = FaceRecon_feat(10, 7)

    def forward(self, pc_list: List[PointCloudPair]):
        points1 = torch.cat([pc.pc1.points.unsqueeze(0) for pc in pc_list], dim=0)  # pc_list is batch size
        points2 = torch.cat([pc.pc2.points.unsqueeze(0) for pc in pc_list], dim=0)  # 3 2000 3
        bs, p_num = points1.shape[0], points1.shape[1]
        feat = (self.face_recon(points1[:,:,0:3] - points1[:,:,0:3].mean(dim=1, keepdim=True))) - (self.face_recon(points2[:,:,0:3] - points2[:,:,0:3].mean(dim=1, keepdim=True)))
        # rotation
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
        

def train(model: test_GPV_Delta, dataloader_train, dataloader_test_inter, dataloader_test_intra, lr, num_epochs, log_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = fs_net_loss_R()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
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

            p_green_R, p_red_R, f_green_R, f_red_R = model(pc_pairs)
            rot1 = ground_truth_rotations([pc.rot_1 for pc in pc_pairs])
            rot2 = ground_truth_rotations([pc.rot_2 for pc in pc_pairs])
            delta_21_rot = calc_delta_rotation(rot1, rot2)
            R_green_gt, R_red_gt = get_gt_v(delta_21_rot)  # Function to get ground truth rotation vectors
            
            pred_list = {
                "Rot1": p_green_R,
                "Rot2": p_red_R,
            }
            gt_list = {
                "Rot1": R_green_gt.cuda(),
                "Rot2": R_red_gt.cuda(),
            }

            sym1, sym2 = get_sym_from_input(pc_pairs)

            loss_dict = criterion(pred_list, gt_list, sym1)
            loss = (loss_dict['Rot1'] + loss_dict['Rot2']) / 2.0
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
            p_green_R, p_red_R, f_green_R, f_red_R = model(pc_pairs)
            rot1 = ground_truth_rotations([pc.rot_1 for pc in pc_pairs])
            rot2 = ground_truth_rotations([pc.rot_2 for pc in pc_pairs])
            delta_21_rot = calc_delta_rotation(rot1, rot2)
            R_green_gt, R_red_gt = get_gt_v(delta_21_rot)  # Function to get ground truth rotation vectors
            
            # Convert predicted vectors and ground truth vectors back to rotation matrices
            pred_rot_matrices = vectors_to_rotation_matrix(p_green_R, p_red_R)
            gt_rot_matrices = vectors_to_rotation_matrix(R_green_gt, R_red_gt)
            
            # Store predictions and ground truths for metrics calculation
            all_pred_rot_matrices.append(pred_rot_matrices.cpu())
            all_gt_rot_matrices.append(gt_rot_matrices.cpu())
    
    all_pred_rot_matrices = torch.cat(all_pred_rot_matrices, dim=0)
    all_gt_rot_matrices = torch.cat(all_gt_rot_matrices, dim=0)

    mean_rot_error = calculate_pose_metrics(
        all_pred_rot_matrices, all_gt_rot_matrices
    )
    writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
    print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    model.train()

if __name__ == "__main__":
    model = test_GPV_Delta().cuda()
    # model.load_state_dict(torch.load('./log_dir/GPV_test/2024-05-22 01:41:18.391449/GPV_[40|40].pth'))
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(root_dir, test_intra_dir, test_inter_dir, voxelization=True, shot=True)
    dataloader_train, dataloader_test_intra, dataloader_test_inter = get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter)
    train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, 0.001, 40, "./log_dir/GPV_delta_test_v2")
    # torch.save(model.state_dict(), "/home/zhangran/desktop/tmp/delta.pth")