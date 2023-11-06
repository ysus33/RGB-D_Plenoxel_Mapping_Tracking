import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from os import path
import imageio


class DatasetBase(Dataset):
    def __init__(self, args, device='cuda:0'):
        super(DatasetBase, self).__init__()
        self.device = device
        self.root = args.data_specs['data_path']
        assert path.isdir(self.root), f"'{self.root}' is not a directory"
        
        self.depth_scale = args.data_specs['depth_scale']
        self.H, self.W, self.fx, self.fy, self.cx, self.cy = args.cam['H'], args.cam[
            'W'], args.cam['fx'], args.cam['fy'], args.cam['cx'], args.cam['cy']

        K = intrinsic_matrix([self.fx, self.fy, self.cx, self.cy]) 

        self.scannet = False
        if args.dataset=='scannet':
            self.scannet = True

        self.range_map_mod = self.getRangeMapModifier()
        
    def getRangeMapModifier(self):
        yy, xx = torch.meshgrid(
            torch.arange(self.H, dtype=torch.float32) + 0.5,
            torch.arange(self.W, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.cx) / self.fx
        yy = (yy - self.cy) / self.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        range_map_mod = torch.norm(dirs, dim=2)

        return range_map_mod.numpy()

    def __len__(self):
        return self.n_img

    def __getitem__(self, index):
        im_path = self.img_paths[index]
        depth_path = self.depth_paths[index]
        
        im_data = imageio.imread(im_path)
        if self.scannet:
            im_data = cv2.resize(im_data, (self.W, self.H))
        im_data = torch.from_numpy(im_data).float()/255.
        
        depth_data = imageio.imread(depth_path)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        depth_data *= self.range_map_mod
        depth_data = torch.from_numpy(depth_data)

        c2w = self.poses[index]
        im_data = im_data.reshape(-1,3) #torch.Size([816000])
        depth_data = depth_data.reshape(-1)

        return index, c2w, im_data, depth_data


    def _get_image(self, index):
        im_path = self.img_paths[index]
        im_data = imageio.imread(im_path)
        if self.scannet:
            im_data = cv2.resize(im_data, (self.W, self.H))

        im_data = im_data.astype(np.float32) / 255.

        depth_path = self.depth_paths[index]
        depth_data = imageio.imread(depth_path)

        depth_data = depth_data.astype(np.float32) / self.depth_scale
        depth_data = torch.from_numpy(depth_data)
        depth_data = self.depth_to_range(depth_data).numpy()

        return im_data, depth_data
   
    def return_cam_params(self):
        return self.H, self.W, self.fx, self.fy, self.cx, self.cy

    def depth_to_range(self, depth_gt):
        height, width = depth_gt.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=self.device) + 0.5,
            torch.arange(width, dtype=torch.float32, device=self.device) + 0.5,
        )
        xx = (xx - self.cx) / self.fx
        yy = (yy - self.cy) / self.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        norm = torch.norm(dirs, dim=2)
        range_gt = depth_gt*norm.cpu()

        return range_gt
        
def intrinsic_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.
    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K