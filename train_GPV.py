import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from network.gpv_layers import *
from datasets.datasets_pair import *
from network.sym_v1 import *
from loss.utils import *
from network.utils import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from loss.losses import fs_net_loss_full
from network.GPVNet import GPVNet
import argparse

def train(model: nn.Module, dataloader_train, dataloader_test_inter, dataloader_test_intra, lr, num_epochs, log_dir):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = fs_net_loss_full()
    name_list = ['Rot1', 'Rot2', 'Rot1_cos', 'Rot2_cos', 'Rot_regular']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # model = DataParallel(model)
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
            test_metrics(model, dataloader_train, device, writer, epoch, 'test_train')
        for batch_idx, batch in enumerate(dataloader_train):
            pc_pairs = [pair.to(device) for pair in batch]
            optimizer.zero_grad()

            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)
            
            # We add transpose to make the rotation matrix. 
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
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

            loss_dict1 = criterion(name_list, pred_list1, gt_list1, sym1)
            loss_dict2 = criterion(name_list, pred_list2, gt_list2, sym2)
            loss = (sum(loss_dict1.values()) + sum(loss_dict2.values())) / 2
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if (batch_idx + 1) % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                print(f"Epoch:[{epoch + 1}|{num_epochs}],Batch:[{(batch_idx + 1)}|{len(dataloader_train)}],Loss:[{loss.item():.4f}]")

        avg_loss = total_loss / len(dataloader_train)
        print(f"Epoch [{epoch+1}|{num_epochs}],Loss:{avg_loss:.4f}")
        writer.add_scalar('train/avg_loss', avg_loss, epoch)

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), log_dir+r'/'+f"GPV_[{epoch+1}|{num_epochs}].pth")
            test_metrics(model, dataloader_test_inter, device, writer, epoch, 'test_inter')
            test_metrics(model, dataloader_test_intra, device, writer, epoch, 'test_intra')
            test_metrics(model, dataloader_train, device, writer, epoch, 'test_train')


def test_metrics(model, dataloader, device, writer, epoch, phase):
    print("______________________" + phase + "_______________________")
    model.eval()
    all_pred_rot_matrices = []
    all_gt_rot_matrices = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            pc_pairs = [pair.to(device) for pair in batch]
            (p_green_R1, p_red_R1, f_green_R1, f_red_R1), (p_green_R2, p_red_R2, f_green_R2, f_red_R2) = model(pc_pairs)
            
            R_green_gt1, R_red_gt1 = get_gt_v(ground_truth_rotations([pc.rot_1.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            R_green_gt2, R_red_gt2 = get_gt_v(ground_truth_rotations([pc.rot_2.T for pc in pc_pairs]))  # Function to get ground truth rotation vectors
            
            # Convert predicted vectors and ground truth vectors back to rotation matrices
            pred_rot_matrices1 = vectors_to_rotation_matrix(p_green_R1, p_red_R1, True)
            pred_rot_matrices2 = vectors_to_rotation_matrix(p_green_R2, p_red_R2, True)
            gt_rot_matrices1 = vectors_to_rotation_matrix(R_green_gt1, R_red_gt1, True)
            gt_rot_matrices2 = vectors_to_rotation_matrix(R_green_gt2, R_red_gt2, True)
            
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
    if writer is not None:
        writer.add_scalar(f'{phase}/mean_rot_error', mean_rot_error, epoch)
    print(f"{phase} - Epoch [{epoch+1}]: Mean Rotation Error: {mean_rot_error:.4f}")
    model.train()

def main():
    args = get_args()
    model = GPVNet().cuda()
    # load the model if needed
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    if args.few_shot:
        args.log_dir += "_few_shot"
    dataset_train, dataset_test_intra, dataset_test_inter = get_datasets(args.root_dir, args.test_intra_dir, args.test_inter_dir, voxelization=False, shot=args.few_shot, choose_category=None, max_points=args.max_points, augmentation=False)
    dataloader_train, dataloader_test_intra, dataloader_test_inter = get_dataloaders(dataset_train, dataset_test_intra, dataset_test_inter, num_workers=0, batch_size=args.batch_size)
    # train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, 0.0001, 300, "./log_dir/GPV_test_new_loss_cat_Camera")
    print("Length of datasets: ")
    print(len(dataset_train),",",len(dataset_test_intra),',',len(dataset_test_inter))
    train(model, dataloader_train, dataloader_test_inter, dataloader_test_intra, args.lr, args.num_epochs, args.log_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/16T/zhangran/GAPartNet_re_rendered/train")
    parser.add_argument("--test_intra_dir", type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_intra")
    parser.add_argument("--test_inter_dir", type=str, default="/16T/zhangran/GAPartNet_re_rendered/test_inter")
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--log_dir", type=str, default="./log_dir/GPVNet")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--no_few_shot', dest='few_shot', action='store_false')
    parser.set_defaults(few_shot=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()