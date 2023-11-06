import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from svox2 import Camera


class Visualizer(object):
    """
    Visualize itermediate results, render out depth, color and depth uncertainty images.
    It can be called per iteration, which is good for debuging (to see how each tracking/mapping iteration performs).

    """
    def __init__(self, vis_dir, slam):
        self.vis_dir = vis_dir
        os.makedirs(f'{vis_dir}', exist_ok=True)
        self.H, self.W, self.fx, self.fy, self.cx, self.cy =\
            slam.H, slam.W, slam.fx, slam.fy, slam.cx, slam.cy
        self.device = slam.device

    def vis(self, grid, idx, gt_color_np, gt_depth_np, c2w):
        with torch.no_grad():
            cam = Camera(c2w.to(device=self.device), 
                            self.fx, self.fy, self.cx, self.cy, self.W, self.H)
            
            # depth_ren = grid.render_image(gt_cam)
            rgb_pred = grid.volume_render_image(cam, use_kernel=True).detach().cpu().numpy()
            depth_pred = grid.volume_render_depth_image(cam).detach().cpu().numpy()
            im_name_r = f'{self.vis_dir}/frame_{idx:05d}_r.jpg'

            self.draw(gt_color_np, gt_depth_np, rgb_pred, depth_pred, im_name_r)

    def draw(self, gt_color_np, gt_depth_np, color_np, depth_np, img_name):
        color_residual = np.abs(gt_color_np - color_np)
        color_residual[gt_depth_np == 0.0] = 0.0
        depth_residual = np.abs(gt_depth_np - depth_np)
        depth_residual[gt_depth_np == 0.0] = 0.0
        max_depth = np.max(gt_depth_np)
        
        fig, axs = plt.subplots(2, 3)
        fig.tight_layout()
        axs[0, 0].imshow(gt_depth_np, cmap="plasma", vmin=0)#, vmax=max_depth)
        axs[0, 0].set_title('Input Depth')
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(depth_np, cmap="plasma", vmin=0)#, vmax=max_depth)

        axs[0, 1].set_title('Generated Depth')
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(depth_residual, cmap="plasma", vmin=0)#, vmax=max_depth)
        axs[0, 2].set_title('Depth Residual')
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        
        gt_color_np = np.clip(gt_color_np, 0, 1)
        color_np = np.clip(color_np, 0, 1)
        color_residual = np.clip(color_residual, 0, 1)
        axs[1, 0].imshow(gt_color_np, cmap="plasma")
        axs[1, 0].set_title('Input RGB')
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(color_np, cmap="plasma")
        axs[1, 1].set_title('Generated RGB')
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(color_residual, cmap="plasma")
        axs[1, 2].set_title('RGB Residual')
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(img_name, bbox_inches='tight', pad_inches=0.2)

        plt.clf()