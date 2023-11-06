import cv2
import sys
import torch
import numpy as np

def rodrigues_vec_to_rotation_mat(rodrigues_vec):
    theta = rodrigues_vec.norm()
    if theta < sys.float_info.epsilon:              
        rotation_mat = torch.eye(3, dtype=torch.float32)
    else:
        r = rodrigues_vec / theta
        I = torch.eye(3, dtype=torch.float32, device=theta.device)
        r_rT = torch.tensor([
            [r[0]*r[0], r[0]*r[1], r[0]*r[2]],
            [r[1]*r[0], r[1]*r[1], r[1]*r[2]],
            [r[2]*r[0], r[2]*r[1], r[2]*r[2]]
        ], device=theta.device)
        r_cross = torch.tensor([
            [0, -r[2], r[1]],
            [r[2], 0, -r[0]],
            [-r[1], r[0], 0]
        ], device=theta.device)
        rotation_mat = torch.cos(theta) * I + (1 - torch.cos(theta)) * r_rT + torch.sin(theta) * r_cross
    return rotation_mat 

def matrix_to_lie(RT):
    """
    Convert transformation matrix to quaternion and translation.

    """
    RT = RT.detach().cpu()              #TODO: Make this more efficent by writing it directly with gpu support
    RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]
    rot,_ = cv2.Rodrigues(R)
    c2w_T = np.concatenate([np.ravel(rot), T], 0)
    c2w_T = torch.from_numpy(c2w_T).float()
    c2w_T = c2w_T
    return c2w_T

def lie_to_matrix(c2w_T): #numpy로 바꾸면 안돼... grad 연결해야함
    rot = rodrigues_vec_to_rotation_mat(c2w_T[:3])
    trans = c2w_T[3:]
    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3,:3] = rot
    c2w[:3, 3] = trans
    return c2w

def quad2rotation(quad, device='cpu'):
    """
    Convert quaternion to rotation in batch. Since all operation in pytorch, support gradient passing.

    Args:
        quad (tensor, batch_size*4): quaternion.

    Returns:
        rot_mat (tensor, batch_size*3*3): rotation.
    """
    bs = quad.shape[0]
    qr, qi, qj, qk = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
    two_s = 2.0 / (quad * quad).sum(-1)
    rot_mat = torch.zeros(bs, 3, 3).to(device)
    rot_mat[:, 0, 0] = 1 - two_s * (qj ** 2 + qk ** 2)
    rot_mat[:, 0, 1] = two_s * (qi * qj - qk * qr)
    rot_mat[:, 0, 2] = two_s * (qi * qk + qj * qr)
    rot_mat[:, 1, 0] = two_s * (qi * qj + qk * qr)
    rot_mat[:, 1, 1] = 1 - two_s * (qi ** 2 + qk ** 2)
    rot_mat[:, 1, 2] = two_s * (qj * qk - qi * qr)
    rot_mat[:, 2, 0] = two_s * (qi * qk - qj * qr)
    rot_mat[:, 2, 1] = two_s * (qj * qk + qi * qr)
    rot_mat[:, 2, 2] = 1 - two_s * (qi ** 2 + qj ** 2)
    return rot_mat

def get_quaternion_from_camera(RT):
    """
    Convert transformation matrix to quaternion and translation.

    """
    gpu_id = -1
    if type(RT) == torch.Tensor:
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    from mathutils import Matrix
    R, T = RT[:3, :3], RT[:3, 3]
    rot = Matrix(R)
    quad = rot.to_quaternion()
    tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor

def get_camera_from_quaternion(inputs, device='cpu'):
    """
    Convert quaternion and translation to transformation matrix.

    """
    N = len(inputs.shape)
    if N == 1:
        inputs = inputs.unsqueeze(0)
    quad, T = inputs[:, :4], inputs[:, 4:]
    R = quad2rotation(quad, device)
    RT = torch.cat([R, T[:, :, None]], 2)
    if N == 1:
        RT = RT[0]
    bottom = torch.from_numpy(np.array([0, 0, 0, 1.]).reshape(
            [1, 4])).type(torch.float32).to(device)
    RT = torch.cat([RT, bottom], dim=0)
    return RT

def get_rays(H, W, fx, fy, cx, cy, c2w, device):
    """
    Get rays for a whole image.
    """
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    # pytorch's meshgrid has indexing='ij'
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
    i = i.t()  # transpose
    j = j.t()
    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(H, W, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)