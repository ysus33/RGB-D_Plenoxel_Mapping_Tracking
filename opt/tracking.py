import gc
import torch
import svox2
from torch.optim.lr_scheduler import StepLR
import cv2
import numpy as np
from util.util_pose_opt import *
from util.criterion import Criterion
from util import config_util
import yaml
import os
from tqdm import tqdm
from util.Visualizer import Visualizer
from dataset.dataset import datasets
from datetime import datetime
from util.util import Timing
from pytictoc import TicToc

class Tracker(object):
    def __init__(self, args):
        self.args = args  # args list
        self.device = args.device
        self.out_dir = os.path.join(
            args.log_dir, args.exp_name)

        assert args.ckpt is not None, "wrong ckpt path"
        self.grid = svox2.SparseGrid.load(args.ckpt, device=self.device)
        self.grid.sh_data.requires_grad_(False)
        self.grid.density_data.requires_grad_(False)

        config_util.setup_render_opts_(self.grid.opt, args.rendering)
        
        self.traj_dir = os.path.join(self.out_dir, 'traj')
        os.makedirs(self.traj_dir, exist_ok=True)

        self.frame_reader = datasets[args.dataset](args=args)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = self.frame_reader.return_cam_params()
        self.n_img = len(self.frame_reader)
        
        self.vis_every = args.vis_every
        self.save_every = args.save_every

        self.cam_lr = args.tracking['cam_lr'] # learning rate of cam pose
        self.seperate_LR = args.tracking['seperate_LR']   # seperate learning rate for rotation and translation
        if self.seperate_LR:
            assert args.tracking['lr_scale_rot'] is not None, \
                "lr_scale_rot must be given under seperate_LR"
            self.lr_scale_rot = args.tracking['lr_scale_rot']
        self.const_speed_assumption = args.tracking['const_speed_assumption']
        self.loss_criteria = Criterion(args)

        self.iter_num = args.tracking['iter_num']
        self.epoch_num = args.tracking['epoch_num']
        self.batch_size = args.tracking['batch_size']

        self.pre_c2w = torch.zeros((4,4))
        self.prepre_c2w = torch.zeros((4,4))

        self.estimate_c2w_list = torch.zeros((self.n_img, 4, 4))
        self.gt_c2w_list = torch.zeros((self.n_img, 4, 4))

        out_path = os.path.join(self.out_dir, "config.yaml")
        yaml.dump(args, open(out_path, 'w'))

        self.visualizer = Visualizer(os.path.join(self.out_dir, 'vis'), self)

    def run(self):
        t = TicToc()
        progress_bar = tqdm(range(0, self.n_img), position=0)
        progress_bar.set_description("tracking frame")
        track_time = []

        for i in progress_bar:
            idx, gt_c2w, im, depth = self.frame_reader[i]

            t.tic()
            # c2w = self.track_frame_check_nonkernel(self.grid, idx, 
            c2w = self.track_frame(self.grid, idx, 
                                       im.to(self.device), 
                                       depth.to(self.device), 
                                       gt_c2w)
            # print(f"OUT TOTAL: {t.tocvalue(restart=True):.4f}")
            self.estimate_c2w_list[idx,:,:] = c2w.clone().detach()
            self.gt_c2w_list[idx,:,:] = gt_c2w
            track_time.append(t.tocvalue(restart=True))
            if idx % self.vis_every == 0:
                gt_img, gt_depth = self.frame_reader._get_image(i)
                self.visualizer.vis(self.grid, idx, gt_img, gt_depth, c2w)
        
            if idx % self.save_every == 0 or idx == self.n_img-1:
                path = os.path.join(self.traj_dir, '{:05d}.tar'.format(idx))
                torch.save({
                    'gt_c2w_list': self.gt_c2w_list,
                    'estimate_c2w_list': self.estimate_c2w_list,
                    'idx': idx,
                }, path, _use_new_zipfile_serialization=False)

        tpath = os.path.join(self.out_dir, 'time_log.txt')
        with open(tpath, 'w') as f:
            for time_ in track_time:
                f.write(f"{time_}\n")

    def create_rays_with_index(self, c2w, index):
        origins = c2w[None, :3, 3].expand(len(index), -1).contiguous()
        yy = torch.floor_divide(index, self.W)
        xx = index - yy*self.W
        xx = (xx + 0.5 - self.cx) / self.fx
        yy = (yy + 0.5 - self.cy) / self.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1).double()
        del xx, yy, zz

        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()
        return svox2.Rays(origins, dirs)

    def track_frame(self, grid, idx, color_gt, depth_gt, gt_c2w):
        if idx == 0:
            self.pre_c2w = gt_c2w
            return gt_c2w
        
        gt_camera_tensor = get_quaternion_from_camera(gt_c2w) 
        if self.const_speed_assumption and idx-2 >= 0:
            pre_c2w = self.pre_c2w.float()
            delta = pre_c2w@self.prepre_c2w.to(pre_c2w.device).float().inverse()
            
            w = torch.norm(delta[:3,:3], p=1) # Manhattan norm of angular velocity (euclidian better?)
            v = torch.norm(delta[:3,3])       # Frobenius norm for linear velocity
            if w+v > 3:      # filter large movement
                init_est_c2w = pre_c2w
            else:
                init_est_c2w = delta@pre_c2w
        else:      
            init_est_c2w = self.pre_c2w
        camera_tensor = get_quaternion_from_camera(init_est_c2w.detach())
        camera_tensor = camera_tensor.to(self.device).requires_grad_()
        cam_para_list = [camera_tensor]
        optimizer_camera = torch.optim.Adam(cam_para_list, lr=self.cam_lr)
        # scheduler = StepLR(optimizer_camera, step_size=5, gamma=0.95)
        scheduler = StepLR(optimizer_camera, step_size=10, gamma=0.8)
        
        initial_cam_tensor_err = gt_camera_tensor.to(self.device)-camera_tensor
        initial_loss_camera_tensor = torch.abs(initial_cam_tensor_err).mean().item()
        initial_ate = torch.sqrt(torch.dot(initial_cam_tensor_err[-3:], initial_cam_tensor_err[-3:]))
        est_cam_tensor = None
        current_min_loss = 10000000000
        self.valid_index = torch.where(depth_gt != 0.0)[0]

        for epoch in range(self.epoch_num):
            perm_id = torch.randint(self.valid_index.size(0), (self.batch_size,), device='cpu')
            index = self.valid_index[perm_id]
                
            c2w = get_camera_from_quaternion(camera_tensor, device=camera_tensor.device)
            color_pred, depth_pred = grid.volume_render_pos_rgbd(
                                                        self.create_rays_with_index(c2w, index))
        
            # with Timing("backward time / iteration"):
            loss, _ = self.loss_criteria(
                                color_pred,
                                depth_pred,
                                color_gt[index],
                                depth_gt[index])
            loss.backward()

            optimizer_camera.step()
            scheduler.step()
            optimizer_camera.zero_grad()

            cam_tensor_err = gt_camera_tensor.to(self.device)-camera_tensor
            loss_camera_tensor = torch.abs(cam_tensor_err).mean().item()
            ate = torch.sqrt(torch.dot(cam_tensor_err[-3:], cam_tensor_err[-3:]))
                
            if epoch == 0:
                start_mae = loss

            if loss < current_min_loss:
                current_min_loss = loss
                min_cam_loss = loss_camera_tensor
                min_ate = ate
                min_d_a = loss.item()
                est_cam_tensor = camera_tensor.clone().detach()

        print(f'ATE: {initial_ate:.6f}->{min_ate:.6f}')

        self.prepre_c2w = self.pre_c2w
        self.pre_c2w = c2w.clone().detach()
        
        torch.cuda.empty_cache()
        
        return c2w