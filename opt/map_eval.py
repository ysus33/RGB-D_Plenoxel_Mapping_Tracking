import math
import argparse
import svox2
import cv2
import numpy as np
from tqdm import tqdm

from util.dataset import datasets
from util import config_util


parser = argparse.ArgumentParser()
parser.add_argument('ckpt', type=str)

config_util.define_common_args(parser)

parser.add_argument('--n_eval', '-n', type=int, default=100000, help='images to evaluate (equal interval), at most evals every image')
parser.add_argument('--train', action='store_true', default=False, help='render train set')
parser.add_argument('--render_path',
                    action='store_true',
                    default=False,
                    help="Render path instead of test images (no metrics will be given)")
parser.add_argument('--timing',
                    action='store_true',
                    default=False,
                    help="Run only for timing (do not save images or use LPIPS/SSIM; "
                    "still computes PSNR to make sure images are being generated)")
parser.add_argument('--no_lpips',
                    action='store_true',
                    default=False,
                    help="Disable LPIPS (faster load)")
parser.add_argument('--no_vid',
                    action='store_true',
                    default=False,
                    help="Disable video generation")
parser.add_argument('--no_imsave',
                    action='store_true',
                    default=False,
                    help="Disable image saving (can still save video; MUCH faster)")
parser.add_argument('--fps',
                    type=int,
                    default=30,
                    help="FPS of video")

# Camera adjustment
parser.add_argument('--crop',
                    type=float,
                    default=1.0,
                    help="Crop (0, 1], 1.0 = full image")

# Include depth comparison
parser.add_argument('--depth_out',
                    action='store_true',
                    default=False,
                    help="Also render a depth comparison")

# Random debugging features
parser.add_argument('--blackbg',
                    action='store_true',
                    default=False,
                    help="Force a black BG (behind BG model) color; useful for debugging 'clouds'")
parser.add_argument('--ray_len',
                    action='store_true',
                    default=False,
                    help="Render the ray lengths")

args = parser.parse_args()
config_util.maybe_merge_config_file(args, allow_invalid=True)
device = 'cuda:0'

def calc_psnr(img, img_gt, avg_psnr):
    mse = (img - img_gt) ** 2
    mse_num : float = mse.mean().item()
    psnr = -10.0 * math.log10(mse_num)
    avg_psnr += psnr
    return avg_psnr


def map_eval(model, dset, n_images):
    config_util.setup_render_opts(model.opt, args)
    n=0
    rgb_avg_psnr = 0.0
    depth_avg_psnr = 0.0
    depth_avg_diff = 0.0

    c2ws = dset.c2w.to(device=device)
    for img_id in tqdm(range(0, n_images)):
        rgb_img_gt = dset.gt[img_id].to(device=device)
        depth_img_gt = dset.depth[img_id].to(device=device)

        dset_h, dset_w = dset.get_image_size(img_id)
        w = dset_w if args.crop == 1.0 else int(dset_w * args.crop)
        h = dset_h if args.crop == 1.0 else int(dset_h * args.crop)
        cam = svox2.Camera(c2ws[img_id],
                    dset.intrins.get('fx', img_id),
                    dset.intrins.get('fy', img_id),
                    dset.intrins.get('cx', img_id) + (w - dset_w) * 0.5,
                    dset.intrins.get('cy', img_id) + (h - dset_h) * 0.5,
                    w, h)

        rgb_img = model.volume_render_image(cam)
        depth_img = model.volume_render_depth_image(cam)

        rgb_img = rgb_img.clamp(0.0,1.0)

        rgb_avg_psnr = calc_psnr(rgb_img, rgb_img_gt, rgb_avg_psnr)
        depth_avg_psnr = calc_psnr(depth_img, depth_img_gt, depth_avg_psnr)

        diff_img = depth_img-depth_img_gt
        diff_img[depth_img_gt == 0.0] = 0.0
        diff_img = diff_img.cpu()
        valid_count = np.count_nonzero(depth_img_gt.cpu() != 0.0)
        depth_diff = np.abs(diff_img).sum()/valid_count
        depth_avg_diff += depth_diff
        print(depth_diff)
        n+=1

    rgb_avg_psnr /= n
    depth_avg_psnr /= n
    depth_avg_diff /= n

    print("RGB PSNR:", rgb_avg_psnr)
    print("Depth PSNR:", depth_avg_psnr)
    print("Depth diff (cm):", depth_avg_diff)


def plenox_eval(args):
    dset = datasets[args.dataset_type](args.data_dir, split="test_train" if args.train else "test",
                                    **config_util.build_data_options(args))
    
    grid = svox2.SparseGrid.load(args.ckpt, device=device)

    n_images = dset.c2w.size(0)
    map_eval(grid, dset, n_images)

plenox_eval(args)