from .dataset_base import DatasetBase
from .util import Rays, Intrin, select_or_shuffle_rays, getRangeMapModifier
from typing import NamedTuple, Optional, Union
from os import path
from tqdm import tqdm
import numpy as np
import imageio
import torch
import json
import glob
import os


class RplcDataset(DatasetBase):

    def __init__(
        self,
        root,
        split,
        epoch_size : Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        scene_scale: Optional[float] = None,
        factor: int = 1,
        scale : Optional[float] = None,
        permutation: bool = True,
        white_bkgd: bool = True,
        n_images = None,
        use_depth = True,
        **kwargs
    ):
        super().__init__()
        self.use_depth = use_depth
        assert path.isdir(root), f"'{root}' is not a directory"

        if scale is None:
            scale = 1.0
        self.device = device
        self.permutation = permutation
        self.epoch_size = epoch_size
        self.use_sphere_bound = False

        all_c2w = []
        all_gt = []
        all_depth = []

        # Load scene config
        scene_name = root.split("/")[-1] + ".json"
        script_dir = os.path.dirname(__file__)
        scene_config_path = os.path.join(script_dir, "../configs/Replica")
        scene_config_path = os.path.join(scene_config_path, scene_name)
        scene_config = json.load(open(scene_config_path, "r"))

        self.scene_radius = scene_config["radius"]
        self.scene_center = scene_config["center"]
        focal, cx, cy = self.loadIntrinParam(scene_config)
        self.intrins_full : Intrin = Intrin(focal, focal, cx, cy)
        range_map_mod = getRangeMapModifier(scene_config["camera"]["h"], scene_config["camera"]["w"],self.intrins_full)

        split_name = split if split != "test_train" else "train"

        data_path = os.path.join(root, "results")
        pose_path = os.path.join(root, "traj.txt")

        print("LOAD DATA", data_path)
        img = sorted(glob.glob(f'{data_path}/frame*.jpg'))
        depth = sorted(glob.glob(f'{data_path}/depth*.png'))
        pose = self.readTxtFile(pose_path)
        self.depth_scale = scene_config['depth_scale']
        assert (len(img) == len(depth)), "gt-D not synced"
        if split_name == "train":
            start = 0
        elif split_name == "test":
            start = 0
        for k in tqdm(range(start, len(img), 10)):
            i = img[k]
            d = depth[k]
            p = pose[k]
            c2w = np.array(list(map(float, p.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            im_gt = imageio.imread(i)
            im_d = imageio.imread(d)
            im_d = im_d.astype(np.float32) * range_map_mod
            all_c2w.append(c2w)
            all_gt.append(torch.from_numpy(im_gt))
            all_depth.append(torch.from_numpy(im_d))
        
        self.c2w = torch.stack(all_c2w)
        self.gt = torch.stack(all_gt).float() / 255.0
        self.depth = torch.stack(all_depth).float() / self.depth_scale
        if self.gt.size(-1) == 4:
            if white_bkgd:
                # Apply alpha channel
                self.gt = self.gt[..., :3] * self.gt[..., 3:] + (1.0 - self.gt[..., 3:])
            else:
                self.gt = self.gt[..., :3]

        self.n_images, self.h_full, self.w_full, _ = self.gt.shape
        if n_images is not None:
            if n_images > self.n_images:
                print(f'using {self.n_images} available training views instead of the requested {n_images}.')
                n_images = self.n_images
            self.n_images = n_images
            self.gt = self.gt[0:n_images,...]
            self.c2w = self.c2w[0:n_images,...]

        self.split = split
        self.scene_scale = scene_scale
        if self.split == "train":
            self.gen_rays(factor=factor)
        else:
            # Rays are not needed for testing
            self.h, self.w = self.h_full, self.w_full
            self.intrins : Intrin = self.intrins_full

    def readTxtFile(self, path):
        file = open(path, mode='r')
        content = file.readlines()
        file.close()
        return content
    
    def loadIntrinParam(self, config):
        focal = config['camera']['fx']
        cx = config['camera']['cx']
        cy = config['camera']['cy']
        return focal, cx, cy
    
    def gen_rays(self, factor=1):
        print(" Generating rays, scaling factor", factor)
        # Generate rays
        self.factor = factor
        self.h = self.h_full // factor
        self.w = self.w_full // factor
        true_factor = self.h_full / self.h
        self.intrins = self.intrins_full.scale(1.0 / true_factor)
        yy, xx = torch.meshgrid(
            torch.arange(self.h, dtype=torch.float32) + 0.5,
            torch.arange(self.w, dtype=torch.float32) + 0.5,
        )
        xx = (xx - self.intrins.cx) / self.intrins.fx
        yy = (yy - self.intrins.cy) / self.intrins.fy
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(1, -1, 3, 1)
        del xx, yy, zz
        dirs = (self.c2w[:, None, :3, :3] @ dirs)[..., 0]

        if factor != 1:
            gt = F.interpolate(
                self.gt.permute([0, 3, 1, 2]), size=(self.h, self.w), mode="area"
            ).permute([0, 2, 3, 1])
            gt = gt.reshape(self.n_images, -1, 3)
        else:
            gt = self.gt.reshape(self.n_images, -1, 3)
        if self.use_depth:
            if factor != 1:
                print("WARNING: Depth not implemented for cases when scaling factor other than 1")
            else:
                depth = self.depth.reshape(self.n_images, -1, 1)
        origins = self.c2w[:, None, :3, 3].expand(-1, self.h * self.w, -1).contiguous()
        if self.split == "train":
            #added invalid depth filtering
            origins = origins.view(-1, 3)
            dirs = dirs.view(-1, 3)
            gt = gt.reshape(-1, 3)
            depth = depth.reshape(-1,1)
            valid_depth = torch.where(depth != 0.0)[0]
            origins = origins[valid_depth]
            dirs = dirs[valid_depth]
            gt = gt[valid_depth]
            depth = depth[valid_depth]

        self.rays_init = Rays(origins=origins, dirs=dirs, gt=gt, depth=depth)
        self.rays = self.rays_init