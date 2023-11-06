# Copyright 2021 Alex Yu
# First, install svox2
# Then, python opt.py <path_to>/nerf_synthetic/<scene> -t ckpt/<some_name>
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, viridis_cmap
from util import config_util

from datetime import datetime
from tensorboardX import SummaryWriter

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)


group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt_d', help='checkpoint and logging directory')

group.add_argument('--reso',
                        type=str,
                        default=
                        "[[256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')
group.add_argument('--upsamp_every', type=int, default=3 * 12800, help='upsample the grid every x iters')
group.add_argument('--init_iters', type=int, default=0, help='do not upsample for first x iters')
group.add_argument('--upsample_density_add', type=float, default=0.0, help='add the remaining density by this amount when upsampling')
group.add_argument('--sh_dim', type=int, default=9, help='SH dimensions (at most 10)')

group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=4 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=5000, help='batch size')


# TODO: make the lr higher near the end
group.add_argument('--lr_sigma', type=float, default=3e1, help='rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)
group.add_argument('--lr_sh', type=float, default=1e-2, help='rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float, default=5e-6)
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)
group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")
group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=1, help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1, help='evaluate every x epochs')
group.add_argument('--init_sigma', type=float, default=0.1, help='initialization sigma')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=True)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None, help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type', choices=["weight", "sigma"], default="weight", help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float, default=0.0005 * 512, help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float, default=5.0, help='Upsample sigma threshold')
group.add_argument('--max_grid_elements', type=int, default=44_000_000,help='Max items to store after upsampling (the number here is given for 22GB memory)')
group.add_argument('--tune_mode', action='store_true', default=False, help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False, help='do not save any checkpoint even at the end')

group = parser.add_argument_group("losses")
# Foreground TV
group.add_argument('--lambda_tv', type=float, default=1e-5)
group.add_argument('--tv_sparsity', type=float, default=0.01)
group.add_argument('--tv_logalpha', action='store_true', default=False, help='Use log(1-exp(-delta * sigma)) as in neural volumes')
group.add_argument('--lambda_tv_sh', type=float, default=1e-3)
group.add_argument('--tv_sh_sparsity', type=float, default=0.01)
group.add_argument('--tv_decay', type=float, default=1.0)
group.add_argument('--tv_early_only', type=int, default=1, help="Turn off TV regularization after the first split/prune")
group.add_argument('--tv_contiguous', type=int, default=1, help="Apply TV only on contiguous link chunks, which is faster")
# End Foreground TV
group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)
group.add_argument('--lr_decay', action='store_true', default=True)
group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               n_images=args.n_train,
               **config_util.build_data_options(args))

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()

grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        radius=dset.scene_radius,
                        use_sphere_bound=False,
                        sh_dim=args.sh_dim,
                        use_z_order=True,
                        device=device)

# DC -> gray; mind the SH scaling!
grid.sh_data.data[:] = 0.0
grid.density_data.data[:] = 0.0 if args.lr_fg_begin_step > 0 else args.init_sigma

grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0]) for i, c2w in enumerate(dset.c2w)
    ]
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0

last_upsamp_step = args.init_iters

epoch_id = -1
while True:
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0])
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                # print(rgb_pred_test.shape)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            img_pred, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                mse_img, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=gstep_id_base, dataformats='HWC')

                rgb_pred_test = rgb_gt_test = None
                mse_num : float = all_mses.mean().item()
                psnr = -10.0 * math.log10(mse_num)
                if math.isnan(psnr):
                    print('NAN PSNR', i, img_id, mse_num)
                    assert False
                stats_test['mse'] += mse_num
                stats_test['psnr'] += psnr
                n_images_gen += 1

            stats_test['mse'] /= n_images_gen
            stats_test['psnr'] /= n_images_gen
            for stat_name in stats_test:
                summary_writer.add_scalar('test/' + stat_name,
                        stats_test[stat_name], global_step=gstep_id_base)
            summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
            print('eval stats:', stats_test)
    if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        # NOTE: we do an eval sanity check, if not in tune_mode
        eval_step()
        gc.collect()

    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            if args.lr_fg_begin_step > 0 and gstep_id == args.lr_fg_begin_step:
                grid.density_data.data[:] = args.init_sigma
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            
            if not args.lr_decay:
                lr_sigma = args.lr_sigma * lr_sigma_factor
                lr_sh = args.lr_sh * lr_sh_factor

            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            depth_gt = dset.rays.depth[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)

            #  with Timing("volrend_fused"):
            rgb_pred, depth_pred = grid.volume_render_fused_rgbd(rays, rgb_gt, depth_gt, d_loss = 1.0)

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)

                if args.weight_decay_sh < 1.0:
                    grid.sh_data.data *= args.weight_decay_sigma
                if args.weight_decay_sigma < 1.0:
                    grid.density_data.data *= args.weight_decay_sh

            # Apply TV/Sparsity regularizers
            if args.lambda_tv > 0.0:
                #  with Timing("tv_inpl"):
                grid.inplace_tv_grad(grid.density_data.grad,
                        scaling=args.lambda_tv,
                        sparse_frac=args.tv_sparsity,
                        logalpha=args.tv_logalpha,
                        contiguous=args.tv_contiguous)
            if args.lambda_tv_sh > 0.0:
                #  with Timing("tv_color_inpl"):
                grid.inplace_tv_color_grad(grid.sh_data.grad,
                        scaling=args.lambda_tv_sh,
                        sparse_frac=args.tv_sh_sparsity,
                        contiguous=args.tv_contiguous)
            # Manual rmsprop step
            if gstep_id >= args.lr_fg_begin_step:
                grid.optim_density_step(lr_sigma, beta=args.rms_beta)
                grid.optim_sh_step(lr_sh, beta=args.rms_beta)
            
    train_step()
    gc.collect()
    gstep_id_base += batches_per_epoch

    if args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode:
        print('Saving', ckpt_path)
        grid.save(ckpt_path)

    if (gstep_id_base - last_upsamp_step) >= args.upsamp_every:
        last_upsamp_step = gstep_id_base
        if reso_id < len(reso_list) - 1:
            print('* Upsampling from', reso_list[reso_id], 'to', reso_list[reso_id + 1])
            if args.tv_early_only > 0:
                print('turning off TV regularization')
                args.lambda_tv = 0.0
                args.lambda_tv_sh = 0.0
            elif args.tv_decay != 1.0:
                args.lambda_tv *= args.tv_decay
                args.lambda_tv_sh *= args.tv_decay

            reso_id += 1
            use_sparsify = True
            z_reso = reso_list[reso_id] if isinstance(reso_list[reso_id], int) else reso_list[reso_id][2]
            grid.resample(reso=reso_list[reso_id],
                    sigma_thresh=args.density_thresh,
                    weight_thresh=args.weight_thresh / z_reso if use_sparsify else 0.0,
                    dilate=2, #use_sparsify,
                    cameras=resample_cameras if args.thresh_type == 'weight' else None,
                    max_elements=args.max_grid_elements)

            if args.upsample_density_add:
                grid.density_data.data[:] += args.upsample_density_add

        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')
        eval_step()
        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"total: {secs / 60} m\n")
        if not args.tune_nosave:
            grid.save(ckpt_path)
        break