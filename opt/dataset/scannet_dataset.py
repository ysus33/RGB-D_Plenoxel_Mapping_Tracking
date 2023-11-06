from .dataset_base import DatasetBase
import os
import glob
import numpy as np
import torch

class ScanNetDataloader(DatasetBase):

    def __init__(self, args, device='cuda:0'):
        super(ScanNetDataloader, self).__init__(args, device)

        data_path = self.root
        trj_path = os.path.join(data_path, "gt.txt")
        self.img_paths = sorted(glob.glob(f'{data_path}/rgb/frame-*.color.jpg'))
        self.n_img = len(self.img_paths)
        self.depth_paths = sorted(glob.glob(f'{data_path}/depth/frame-*.depth.pgm'))

        self.load_poses(trj_path)


    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(3, 4)
            c2w = np.vstack((c2w, np.array([0,0,0,1])))
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)