import torch

def get_gt_v(Rs, axis=2):
    bs = Rs.shape[0]  # bs x 3 x 3
    # TODO use 3 axis, the order remains: do we need to change order?
    if axis == 3:
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    else:
        assert axis == 2
        corners = torch.tensor([[0, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=torch.float).to(Rs.device)
        corners = corners.view(1, 3, 3).repeat(bs, 1, 1)  # bs x 3 x 3
        gt_vec = torch.bmm(Rs, corners).transpose(2, 1).reshape(bs, -1)
    gt_green = gt_vec[:, 3:6]
    gt_red = gt_vec[:, (6, 7, 8)]
    return gt_green, gt_red

def vectors_to_rotation_matrix(green_vector, red_vector, transpose = False):
    # green_vector and red_vector are normalized
    green_vector = green_vector / torch.norm(green_vector, dim=1, keepdim=True)
    red_vector = red_vector / torch.norm(red_vector, dim=1, keepdim=True)
    blue_vector = torch.cross(green_vector, red_vector)
    if transpose:
        blue_vector = - blue_vector
    
    rotation_matrix = torch.stack([red_vector, green_vector, blue_vector], dim=2)
    return rotation_matrix

def calc_delta_rotation(rot_1: torch.Tensor, rot_2: torch.Tensor) -> torch.Tensor:
    """
    input: two of rotation (bs, 3, 3) 
    output: delta_rot_21 of rot_1 and rot_2, rot_2 = delta_rot_21 @ rot_1 (mutiply)
    """
    assert rot_1.shape == rot_2.shape and rot_1.shape[-2:] == (3, 3), "Input rotations must be of shape (bs, 3, 3)"
    rot_1_inv = rot_1.transpose(-1, -2)
    delta_rot_21 = torch.matmul(rot_2, rot_1_inv)
    return delta_rot_21
