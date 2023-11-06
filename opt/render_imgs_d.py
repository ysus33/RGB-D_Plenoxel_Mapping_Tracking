# Copyright 2021 Alex Yu
# Eval

import torch
import svox2
import svox2.utils
import math
import argparse
import numpy as np
import os
from os import path
from util.dataset import datasets
from util.util import viridis_cmap
from util import config_util

import imageio
import cv2
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

if not path.isfile(args.ckpt):
    args.ckpt = path.join(args.ckpt, 'ckpt.npz')

render_dir = path.join(path.dirname(args.ckpt), 'test_renders_depth')
dset = datasets[args.dataset_type](args.data_dir, split="test", **config_util.build_data_options(args))

grid = svox2.SparseGrid.load(args.ckpt, device=device)
config_util.setup_render_opts(grid.opt, args)

print('Writing to', render_dir)
os.makedirs(render_dir, exist_ok=True)

# with torch.no_grad():
#     n_images = dset.n_images
#     img_eval_interval = max(n_images // args.n_eval, 1)
#     c2ws = dset.c2w.to(device=device)
#     frames = []

#     for img_id in tqdm(range(0, n_images, img_eval_interval)):
#         h, w = dset.get_image_size(img_id)
#         cam = svox2.Camera(c2ws[img_id],
#                            dset.intrins.get('fx', img_id),
#                            dset.intrins.get('fy', img_id),
#                            dset.intrins.get('cx', img_id) + 0.5,
#                            dset.intrins.get('cy', img_id) + 0.5,
#                            w, h)
#         im = grid.volume_render_depth_image(cam)
      
#         img_path = path.join(render_dir, f'depth_{img_id:04d}.png');
#         im = im.cpu().numpy()
#         im_gt = dset.depth[img_id].numpy()
#         m = np.max(im)
#         im_gt = viridis_cmap(im_gt)
#         im = im / m
#         im = np.clip(im, 0, 1)
#         im = viridis_cmap(im)
#         im = np.concatenate([im_gt, im], axis=1)
#         imageio.imwrite(img_path,im)
#         im = None


with torch.no_grad():
    n_images = dset.n_images
    img_eval_interval = max(n_images // args.n_eval, 1)
    c2ws = dset.c2w.to(device=device)
    frames = []

    for img_id in tqdm(range(0, n_images, img_eval_interval)):
        h, w = dset.get_image_size(img_id)
        cam = svox2.Camera(c2ws[img_id],
                           dset.intrins.get('fx', img_id),
                           dset.intrins.get('fy', img_id),
                           dset.intrins.get('cx', img_id) + 0.5,
                           dset.intrins.get('cy', img_id) + 0.5,
                           w, h)
        im = grid.volume_render_depth_image(cam)
      
        im = im.cpu().numpy()
        im_gt = dset.depth[img_id].numpy()
        m = np.max(im)
        im_gt = viridis_cmap(im_gt)
        im = im / m
        im = np.clip(im, 0, 1)
        im = viridis_cmap(im)
        im = np.concatenate([im_gt, im], axis=1)
        frames.append(im)
        im = None

    vid_path = render_dir + '.mp4'
    imageio.mimwrite(vid_path, frames, fps=10, macro_block_size=8)  # pip install imageio-ffmpeg

