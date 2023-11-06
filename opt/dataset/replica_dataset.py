from .dataset_base import DatasetBase
from os import path
import numpy as np
import torch
import glob


class ReplicaDataloader(DatasetBase):
    def __init__(self, args, device='cuda:0'):
        super(ReplicaDataloader, self).__init__(args, device)
        
        data_path = path.join(self.root, "results")
        trj_path = path.join(self.root, "traj.txt")
        self.img_paths = sorted(glob.glob(f'{data_path}/frame*.jpg'))
        self.n_img = len(self.img_paths)
        self.depth_paths = sorted(glob.glob(f'{data_path}/depth*.png'))
        
        self.load_poses(trj_path)

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)
