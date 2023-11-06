import torch
import torch.cuda
from typing import Optional, Union, List
from dataclasses import dataclass
import numpy as np
import cv2
from matplotlib import pyplot as plt


@dataclass
class Rays:
    origins: Union[torch.Tensor, List[torch.Tensor]]
    dirs: Union[torch.Tensor, List[torch.Tensor]]
    gt: Union[torch.Tensor, List[torch.Tensor]]
    depth: Union[torch.Tensor, List[torch.Tensor]] = None

    def to(self, *args, **kwargs):
        origins = self.origins.to(*args, **kwargs)
        dirs = self.dirs.to(*args, **kwargs)
        gt = self.gt.to(*args, **kwargs)
        if self.depth != None:
            depth = self.depth.to(*args, **kwargs)
        else:
            depth = None
        return Rays(origins, dirs, gt, depth)

    def __getitem__(self, key):
        origins = self.origins[key]
        dirs = self.dirs[key]
        gt = self.gt[key]
        if self.depth != None:
            depth = self.depth[key]
        else:
            depth = None
        return Rays(origins, dirs, gt, depth)

    def __len__(self):
        return self.origins.size(0)

@dataclass
class Intrin:
    fx: Union[float, torch.Tensor]
    fy: Union[float, torch.Tensor]
    cx: Union[float, torch.Tensor]
    cy: Union[float, torch.Tensor]

    def scale(self, scaling: float):
        return Intrin(
                    self.fx * scaling,
                    self.fy * scaling,
                    self.cx * scaling,
                    self.cy * scaling
                )

    def get(self, field:str, image_id:int=0):
        val = self.__dict__[field]
        return val if isinstance(val, float) else val[image_id].item()


class Timing:
    """
    Timing environment
    usage:
    with Timing("message"):
        your commands here
    will print CUDA runtime in ms
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()

    def __exit__(self, type, value, traceback):
        self.end.record()
        torch.cuda.synchronize()
        print(self.name, "elapsed", self.start.elapsed_time(self.end), "ms")


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Continuous learning rate decay function. Adapted from JaxNeRF

    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.

    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def viridis_cmap(gray: np.ndarray):
    """
    Visualize a single-channel image using matplotlib's viridis color map
    yellow is high value, blue is low
    :param gray: np.ndarray, (H, W) or (H, W, 1) unscaled
    :return: (H, W, 3) float32 in [0, 1]
    """
    colored = plt.cm.viridis(plt.Normalize()(gray.squeeze()))[..., :-1]
    return colored.astype(np.float32)

def save_img(img: np.ndarray, path: str):
    """Save an image to disk. Image should have values in [0,1]."""
    img = np.array((np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def select_or_shuffle_rays(rays_init : Rays,
                 permutation: int = False,
                 epoch_size: Optional[int] = None,
                 device: Union[str, torch.device] = "cpu"):
    n_rays = rays_init.origins.size(0)
    n_samp = n_rays if (epoch_size is None) else epoch_size
    if permutation:
        print(" Shuffling rays")
        indexer = torch.randperm(n_rays, device='cpu')[:n_samp]
    else:
        print(" Selecting random rays")
        indexer = torch.randint(n_rays, (n_samp,), device='cpu')
    return rays_init[indexer].to(device=device)

def getRangeMapModifier(height, width, intrinsic):
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
    )
    xx = (xx - intrinsic.cx) / intrinsic.fx
    yy = (yy - intrinsic.cy) / intrinsic.fy
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
    del xx, yy, zz
    range_map_mod = torch.norm(dirs, dim=2)

    return range_map_mod.numpy()