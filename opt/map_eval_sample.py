import math
import argparse
import svox2
import cv2
import numpy as np
from tqdm import tqdm

from dataset.dataset import datasets
from parser import get_parser
from util import config_util


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)
parser.add_argument('--n_eval', type=int, default=100, help='sample frame number')
parser.add_argument('--n_ray', type=int, default=10000, help='sample ray number per frame')


args = parser.parse_args()
device = 'cuda:0'

def calc_psnr(img, img_gt, avg_psnr):
    mse = (img - img_gt) ** 2
    mse_num : float = mse.mean().item()
    psnr = -10.0 * math.log10(mse_num)
    avg_psnr += psnr
    return avg_psnr

def random_select(l, k):
    """
    Random select k values from 0..l.

    """
    return list(np.random.permutation(np.array(range(l)))[:min(l, k)])

def map_eval(model, dset, n_images):
    rgb_avg_psnr = 0.0
    depth_avg_psnr = 0.0
    depth_avg_diff = 0.0

    samp_frames = random_select(n_images, args.n_eval)
    h, w, fx, fy, cx, cy = dset.return_cam_params()

    for img_id in tqdm(samp_frames):
        samp_pix = random_select(h*w, args.n_ray)
        _, gt_c2w, rgb_img_gt, depth_img_gt = dset[img_id]

        rgb_img_gt = rgb_img_gt[samp_pix].to(device=device)
        depth_img_gt = depth_img_gt[samp_pix].to(device=device)
        cam = svox2.Camera(gt_c2w.cuda(),
                    fx,
                    fy,
                    cx,
                    cy,
                    w, h)

        rgb_img = model.volume_render_image(cam)
        rgb_img = rgb_img.reshape(h*w, 3)[samp_pix]
        depth_img = model.volume_render_depth_image(cam)
        depth_img = depth_img.reshape(h*w)[samp_pix]
        rgb_img = rgb_img.clamp(0.0,1.0)

        mask = [depth_img_gt != 0.0]
        rgb_avg_psnr = calc_psnr(rgb_img[mask], rgb_img_gt[mask], rgb_avg_psnr)

        diff_img = depth_img-depth_img_gt
        diff_img[depth_img_gt == 0.0] = 0.0
        valid_count = np.count_nonzero(depth_img_gt.cpu() != 0.0)
        diff_img = diff_img.cpu()
        depth_diff = np.abs(diff_img).sum()/valid_count
        depth_avg_diff += depth_diff

    print("RGB PSNR:", rgb_avg_psnr/args.n_eval)
    print("Depth diff (m):", depth_avg_diff/args.n_eval)


def plenox_eval(args):
    args = get_parser().parse_args()
    # args.dataset_type = 'scannet'
    args.dataset_type = 'replica'
    dset = datasets[args.dataset_type](args=args)
    
    grid = svox2.SparseGrid.load(args.ckpt, device=device)
    config_util.setup_render_opts_(grid.opt, args.rendering)

    n_images = len(dset)
    map_eval(grid, dset, n_images)

plenox_eval(args)