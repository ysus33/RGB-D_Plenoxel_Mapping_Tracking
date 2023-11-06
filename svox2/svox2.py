from . import utils
import torch
from torch import nn, autograd
import torch.nn.functional as F
from typing import Union, List, Optional, Tuple
from dataclasses import dataclass
from warnings import warn
from functools import reduce
from tqdm import tqdm
import numpy as np
import svox2.csrc as _C

# _C = utils._get_c_extension()


@dataclass
class RenderOptions:
    """
    Rendering options, see comments
    """
    empty_space_brightness: float = 1.0  # [0, 1], the background color black-white
    step_size: float = 0.5  # Step size, in normalized voxels (not used for svox1)
    sigma_thresh: float = 1e-10  # Voxels with sigmas < this are ignored, in [0, 1]
    stop_thresh: float = 1e-7  # Stops rendering if the remaining light intensity/termination, in [0, 1]
    last_sample_opaque: bool = False   # Make the last sample opaque (for forward-facing)
    near_clip: float = 0.0
    use_spheric_clip: bool = False

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        opt = _C.RenderOptions()
        opt.empty_space_brightness = self.empty_space_brightness
        opt.step_size = self.step_size
        opt.sigma_thresh = self.sigma_thresh
        opt.stop_thresh = self.stop_thresh
        opt.near_clip = self.near_clip
        opt.use_spheric_clip = self.use_spheric_clip
        opt.last_sample_opaque = self.last_sample_opaque
        return opt


@dataclass
class Rays:
    origins: torch.Tensor
    dirs: torch.Tensor

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.RaysSpec()
        spec.origins = self.origins
        spec.dirs = self.dirs
        return spec

    def __getitem__(self, key):
        return Rays(self.origins[key], self.dirs[key])

    @property
    def is_cuda(self) -> bool:
        return self.origins.is_cuda and self.dirs.is_cuda


@dataclass
class Camera:
    c2w: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    def _to_cpp(self):
        """
        Generate object to pass to C++
        """
        spec = _C.CameraSpec()
        spec.c2w = self.c2w
        spec.fx = self.fx_val
        spec.fy = self.fy_val
        spec.cx = self.cx_val
        spec.cy = self.cy_val
        spec.width = self.width
        spec.height = self.height
        return spec

    @property
    def is_cuda(self) -> bool:
        return self.c2w.is_cuda

    def gen_rays(self) -> Rays:
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device) + 0.5,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        return Rays(origins, dirs)
    
    def gen_rays_(self):
        """
        Generate the rays for this camera
        :return: (origins (H*W, 3), dirs (H*W, 3))
        """
        origins = self.c2w[None, :3, 3].expand(self.height * self.width, -1).contiguous()
        yy, xx = torch.meshgrid(
            torch.arange(self.height, dtype=torch.float64, device=self.c2w.device) + 0.5,
            torch.arange(self.width, dtype=torch.float64, device=self.c2w.device) + 0.5,
        )
        xx = (xx - self.cx_val) / self.fx_val
        yy = (yy - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()

        return origins, dirs

    def create_rays_with_index(self, index):
        origins = self.c2w[None, :3, 3].expand(len(index), -1).contiguous()
        yy = torch.floor_divide(index, self.width)
        xx = index - yy*self.width
        xx = (xx + 0.5 - self.cx_val) / self.fx_val
        yy = (yy + 0.5 - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1).double()
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()
        return origins, dirs

@dataclass
class qCamera:
    qtensor: torch.Tensor  # OpenCV
    fx: float = 1111.11
    fy: Optional[float] = None
    cx: Optional[float] = None
    cy: Optional[float] = None
    width: int = 800
    height: int = 800

    @property
    def fx_val(self):
        return self.fx

    @property
    def fy_val(self):
        return self.fx if self.fy is None else self.fy

    @property
    def cx_val(self):
        return self.width * 0.5 if self.cx is None else self.cx

    @property
    def cy_val(self):
        return self.height * 0.5 if self.cy is None else self.cy

    def create_rays_with_index(self, index):
        origins = self.c2w[None, :3, 3].expand(len(index), -1).contiguous()
        yy = torch.floor_divide(index, self.width)
        xx = index - yy*self.width
        xx = (xx + 0.5 - self.cx_val) / self.fx_val
        yy = (yy + 0.5 - self.cy_val) / self.fy_val
        zz = torch.ones_like(xx)
        dirs = torch.stack((xx, yy, zz), dim=-1).double()
        del xx, yy, zz
        dirs /= torch.norm(dirs, dim=-1, keepdim=True)
        dirs = dirs.reshape(-1, 3, 1)
        dirs = (self.c2w[None, :3, :3].double() @ dirs)[..., 0]
        dirs = dirs.reshape(-1, 3).float()
        return origins, dirs

# BEGIN Differentiable CUDA functions with custom gradient
class _SampleGridAutogradFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        grid,
        points: torch.Tensor,
        want_colors: bool,
    ):
        assert not points.requires_grad, "Point gradient not supported"
        out_density, out_sh = _C.sample_grid(grid, points, want_colors)
        ctx.save_for_backward(points)
        ctx.grid = grid
        ctx.want_colors = want_colors
        return out_density, out_sh

    @staticmethod
    def backward(ctx, grad_out_density, grad_out_sh):
        (points,) = ctx.saved_tensors
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)
        _C.sample_grid_backward(
            ctx.grid,
            points,
            grad_out_density.contiguous(),
            grad_out_sh.contiguous(),
            grad_density_grid,
            grad_sh_grid,
            ctx.want_colors
        )
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None

        return grad_density_grid, grad_sh_grid, None, None, None

class _VolumeRenderFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data_density: torch.Tensor,
        data_sh: torch.Tensor,
        grid,
        rays,
        opt,
    ):
        cu_fn = _C.__dict__[f"volume_render_cuvol"]
        color = cu_fn(grid, rays, opt)
        ctx.save_for_backward(color)
        ctx.grid = grid
        ctx.rays = rays
        ctx.opt = opt
        return color

    @staticmethod
    def backward(ctx, grad_out):
        (color_cache,) = ctx.saved_tensors
        cu_fn = _C.__dict__[f"volume_render_cuvol_backward"]
        grad_density_grid = torch.zeros_like(ctx.grid.density_data.data)
        grad_sh_grid = torch.zeros_like(ctx.grid.sh_data.data)

        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density_grid
        grad_holder.grad_sh_out = grad_sh_grid
        cu_fn(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_out.contiguous(),
            color_cache,
            grad_holder
        )
        ctx.grid = ctx.rays = ctx.opt = None
        if not ctx.needs_input_grad[0]:
            grad_density_grid = None
        if not ctx.needs_input_grad[1]:
            grad_sh_grid = None

        return grad_density_grid, grad_sh_grid, None, None, None

class _RenderFunctionPoseRGB(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        grid,
        opt
    ):
        color = torch.empty((origins.size(0),3), 
                             dtype=torch.float32, 
                             device=origins.device)
        rays = Rays(origins, dirs)._to_cpp()
        log_trans = _C.render_pos_forward(grid, rays, opt, color)
        ctx.save_for_backward(color)
        ctx.origins = origins
        ctx.dirs = dirs
        ctx.grid = grid
        ctx.opt = opt
        ctx.rays = rays
        return color

    @staticmethod
    def backward(ctx, grad_color):
        (color_cache, ) = ctx.saved_tensors
        grad_origin = torch.zeros_like(ctx.origins.data)
        grad_dir = torch.zeros_like(ctx.dirs.data)
        
        grad_holder = _C.PoseGrads()
        grad_holder.grad_pose_origin_out = grad_origin
        grad_holder.grad_pose_direction_out = grad_dir

        _C.render_pos_backward(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_color.contiguous(),
            color_cache,
            grad_holder)
        ctx.grid = ctx.opt = None
        torch.cuda.synchronize()
        
        return grad_origin, grad_dir, None, None, None
    
class _RenderFunctionPoseRGBD(autograd.Function):
    @staticmethod
    def forward( 
        ctx,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        grid,
        opt
    ):
        color = torch.empty((origins.size(0),3), 
                             dtype=torch.float32, 
                             device=origins.device)
        
        depth = torch.empty((origins.size(0),), 
                             dtype=torch.float32, 
                             device=origins.device)
        
        rays = Rays(origins, dirs)._to_cpp()
        log_trans = _C.render_pos_forward_rgbd(grid, rays, opt, color, depth)
        ctx.save_for_backward(color, depth)
        ctx.origins = origins
        ctx.dirs = dirs
        ctx.grid = grid
        ctx.opt = opt
        ctx.rays = rays
        return color, depth

    @staticmethod
    def backward(ctx, grad_color, grad_depth):
        color_cache, depth_cache = ctx.saved_tensors
        grad_origin = torch.zeros_like(ctx.origins.data)
        grad_dir = torch.zeros_like(ctx.dirs.data)
        
        grad_holder = _C.PoseGrads()
        grad_holder.grad_pose_origin_out = grad_origin
        grad_holder.grad_pose_direction_out = grad_dir

        _C.render_pos_backward_rgbd(
            ctx.grid,
            ctx.rays,
            ctx.opt,
            grad_color.contiguous(),
            grad_depth.contiguous(),
            color_cache,
            depth_cache,
            grad_holder)
        ctx.grid = ctx.opt = None
        torch.cuda.synchronize()
        
        return grad_origin, grad_dir, None, None, None

class _TotalVariationFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        data: torch.Tensor,
        links: torch.Tensor,
        start_dim: int,
        end_dim: int,
        use_logalpha: bool,
        logalpha_delta: float,
        ignore_edge: bool,
    ):
        tv = _C.tv(links, data, start_dim, end_dim, use_logalpha, logalpha_delta, ignore_edge)
        ctx.save_for_backward(links, data)
        ctx.start_dim = start_dim
        ctx.end_dim = end_dim
        ctx.use_logalpha = use_logalpha
        ctx.logalpha_delta = logalpha_delta
        ctx.ignore_edge = ignore_edge
        return tv

    @staticmethod
    def backward(ctx, grad_out):
        links, data = ctx.saved_tensors
        grad_grid = torch.zeros_like(data)
        _C.tv_grad(links, data, ctx.start_dim, ctx.end_dim, 1.0,
                   ctx.use_logalpha, ctx.logalpha_delta,
                   ctx.ignore_edge,
                   grad_grid)
        ctx.start_dim = ctx.end_dim = None
        if not ctx.needs_input_grad[0]:
            grad_grid = None
        return grad_grid, None, None, None,\
               None, None, None, None


# END Differentiable CUDA functions with custom gradient

class SparseGrid(nn.Module):
    """
    Main sparse grid data structure.
    initially it will be a dense grid of resolution <reso>.
    Only float32 is supported.

    :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
    :param radius: float or List[float, float, float], the 1/2 side length of the grid, optionally in each direction
    :param center: float or List[float, float, float], the center of the grid
    :param sh_dim: int, number of SH components (must be square number)
    :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
    :param device: torch.device, device to store the grid
    """

    def __init__(
        self,
        reso: Union[int, List[int], Tuple[int, int, int]] = 128,
        radius: Union[float, List[float]] = 1.0,
        center: Union[float, List[float]] = [0.0, 0.0, 0.0],
        sh_dim: int = 9,  # SH number, square number
        use_z_order : bool=False,
        use_sphere_bound : bool=False,
        mlp_width : int = 16,
        device: Union[torch.device, str] = "cpu",
    ):
        super().__init__()
        assert (
            sh_dim >= 1 and sh_dim <= utils.MAX_SH_BASIS
        ), f"sh_dim 1-{utils.MAX_SH_BASIS} supported"
        self.sh_dim = sh_dim
        if isinstance(reso, int):
            reso = [reso] * 3
        else:
            assert (
                len(reso) == 3
            ), "reso must be an integer or indexable object of 3 ints"

        if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
            print("Morton code requires a cube grid of power-of-2 size, ignoring...")
            use_z_order = False

        if isinstance(radius, float) or isinstance(radius, int):
            radius = [radius] * 3
        if isinstance(radius, torch.Tensor):
            radius = radius.to(device="cpu", dtype=torch.float32)
        else:
            radius = torch.tensor(radius, dtype=torch.float32, device="cpu")
        if isinstance(center, torch.Tensor):
            center = center.to(device="cpu", dtype=torch.float32)
        else:
            center = torch.tensor(center, dtype=torch.float32, device="cpu")

        self.radius: torch.Tensor = radius  # CPU
        self.center: torch.Tensor = center  # CPU
        self._offset = 0.5 * (1.0 - self.center / self.radius)
        self._scaling = 0.5 / self.radius

        n3: int = reduce(lambda x, y: x * y, reso)
        if use_z_order:
            init_links = utils.gen_morton(reso[0], device=device, dtype=torch.int32).flatten()
        else:
            init_links = torch.arange(n3, device=device, dtype=torch.int32)

        if use_sphere_bound:
            X = torch.arange(reso[0], dtype=torch.float32, device=device) - 0.5
            Y = torch.arange(reso[1], dtype=torch.float32, device=device) - 0.5
            Z = torch.arange(reso[2], dtype=torch.float32, device=device) - 0.5
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)
            gsz = torch.tensor(reso)
            roffset = 1.0 / gsz - 1.0
            rscaling = 2.0 / gsz
            points = torch.addcmul(
                roffset.to(device=points.device),
                points,
                rscaling.to(device=points.device),
            )

            norms = points.norm(dim=-1)
            mask = norms <= 1.0 + (3 ** 0.5) / gsz.max()
            self.capacity: int = mask.sum()

            data_mask = torch.zeros(n3, dtype=torch.int32, device=device)
            idxs = init_links[mask].long()
            data_mask[idxs] = 1
            data_mask = torch.cumsum(data_mask, dim=0) - 1

            init_links[mask] = data_mask[idxs].int()
            init_links[~mask] = -1
        else:
            self.capacity = n3

        self.density_data = nn.Parameter(
            torch.zeros(self.capacity, 1, dtype=torch.float32, device=device)
        )
        self.sh_data = nn.Parameter(
            torch.zeros(
                self.capacity, self.sh_dim * 3, dtype=torch.float32, device=device
            )
        )

        self.register_buffer("links", init_links.view(reso))
        self.links: torch.Tensor
        self.opt = RenderOptions()
        self.sparse_grad_indexer: Optional[torch.Tensor] = None
        self.sparse_sh_grad_indexer: Optional[torch.Tensor] = None
        self.density_rms: Optional[torch.Tensor] = None
        self.sh_rms: Optional[torch.Tensor] = None

        if self.links.is_cuda and use_sphere_bound:
            self.accelerate()

    @property
    def data_dim(self):
        """
        Get the number of channels in the data, including color + density
        (similar to svox 1)
        """
        return self.sh_data.size(1) + 1

    @property
    def shape(self):
        return list(self.links.shape) + [self.data_dim]

    def _fetch_links(self, links):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32
        )
        results_sh = torch.zeros(
            (links.size(0), self.sh_data.size(1)),
            device=links.device,
            dtype=torch.float32,
        )
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        results_sh[mask] = self.sh_data[idxs]
        return results_sigma, results_sh
    
    def _fetch_links_sigma(self, links):
        results_sigma = torch.zeros(
            (links.size(0), 1), device=links.device, dtype=torch.float32
        )
       
        mask = links >= 0
        idxs = links[mask].long()
        results_sigma[mask] = self.density_data[idxs]
        return results_sigma

    def sample(self, points: torch.Tensor,
               use_kernel: bool = True,
               grid_coords: bool = False,
               want_colors: bool = True):
        """
        Grid sampling with trilinear interpolation.
        Behaves like torch.nn.functional.grid_sample
        with padding mode border and align_corners=False (better for multi-resolution).

        Any voxel with link < 0 (empty) is considered to have 0 values in all channels
        prior to interpolating.

        :param points: torch.Tensor, (N, 3)
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param grid_coords: bool, if true then uses grid coordinates ([-0.5, reso[i]-0.5 ] in each dimension);
                                  more numerically exact for resampling
        :param want_colors: bool, if true (default) returns density and colors,
                            else returns density and a dummy tensor to be ignored
                            (much faster)

        :return: (density, color)
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            assert points.is_cuda
            return _SampleGridAutogradFunction.apply(
                self.density_data, self.sh_data, self._to_cpp(grid_coords=grid_coords), points, want_colors
            )
        else:
            if not grid_coords:
                points = self.world2grid(points)
            points.clamp_min_(0.0)
            for i in range(3):
                points[:, i].clamp_max_(self.links.size(i) - 1)
            l = points.to(torch.long)
            for i in range(3):
                l[:, i].clamp_max_(self.links.size(i) - 2)
            wb = points - l
            wa = 1.0 - wb

            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            samples_sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            if want_colors:
                c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
                c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
                c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
                c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
                c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
                c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
                samples_rgb = c0 * wa[:, :1] + c1 * wb[:, :1]
            else:
                samples_rgb = torch.empty_like(self.sh_data[:0])

            return samples_sigma, samples_rgb

    def forward(self, points: torch.Tensor, use_kernel: bool = True):
        return self.sample(points, use_kernel=use_kernel)
    
    def _volume_render_rgbd_pytorch(self, rays: Rays):
        """
        trilerp gradcheck version of RGBD
        """
        self.sh_data.requires_grad_(False)
        self.density_data.requires_grad_(False)

        origins = self.world2grid(rays.origins)
        dirs = rays.dirs / torch.norm(rays.dirs, dim=-1, keepdim=True)

        viewdirs = dirs
        B = dirs.size(0)
        assert origins.size(0) == B
        gsz = self._grid_size()
        dirs = dirs * (self._scaling * gsz).to(device=dirs.device)
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs = dirs * delta_scale.unsqueeze(-1)
        
        sh_mult = utils.eval_sh_bases(self.sh_dim, viewdirs)
        invdirs = 1.0 / dirs

        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        out_depth = torch.zeros((B, ), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0].clone(), gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1].clone(), gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2].clone(), gsz_cu[2] - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0].clone(), gsz_cu[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1].clone(), gsz_cu[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2].clone(), gsz_cu[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = self._fetch_links(links000)
            sigma001, rgb001 = self._fetch_links(links001)
            sigma010, rgb010 = self._fetch_links(links010)
            sigma011, rgb011 = self._fetch_links(links011)
            sigma100, rgb100 = self._fetch_links(links100)
            sigma101, rgb101 = self._fetch_links(links101)
            sigma110, rgb110 = self._fetch_links(links110)
            sigma111, rgb111 = self._fetch_links(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, self.sh_dim)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :3].clone()
            depth = weight * t * delta_scale[good_indices] 

            out_rgb[good_indices] = out_rgb[good_indices] + rgb
            out_depth[good_indices] = out_depth[good_indices] + depth

            log_light_intensity[good_indices] = log_light_intensity[good_indices] + log_att
            t = t + self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask].clone()
            dirs = dirs[mask].clone()
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]

        if self.opt.empty_space_brightness:
            out_rgb = out_rgb + (
                torch.exp(log_light_intensity).unsqueeze(-1)
                * self.opt.empty_space_brightness
            )

        return out_rgb, out_depth

    def _volume_render_depth_pytorch(self, rays: Rays):
        """
        trilerp gradcheck of DEPTH
        """
        self.density_data.requires_grad_(False)

        origins = self.world2grid(rays.origins)
        B = origins.size(0)
        gsz = self._grid_size()
        delta_scale = 1./((self._scaling * gsz).to(device=rays.dirs.device))
        dirs = rays.dirs
        invdirs = 1.0 / dirs.clone().detach()

        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(self.opt.near_clip)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_depth = torch.zeros((B, ), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)
        del invdirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        t = t[mask]
        tmax = tmax[mask]

        del t1, t2

        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0].clone(), gsz_cu[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1].clone(), gsz_cu[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2].clone(), gsz_cu[2] - 1)
            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0].clone(), gsz_cu[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1].clone(), gsz_cu[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2].clone(), gsz_cu[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)

            del l

            links000 = self.links[lx, ly, lz]
            links001 = self.links[lx, ly, lz + 1]
            links010 = self.links[lx, ly + 1, lz]
            links011 = self.links[lx, ly + 1, lz + 1]
            links100 = self.links[lx + 1, ly, lz]
            links101 = self.links[lx + 1, ly, lz + 1]
            links110 = self.links[lx + 1, ly + 1, lz]
            links111 = self.links[lx + 1, ly + 1, lz + 1]

            sigma000 = self._fetch_links_sigma(links000)
            sigma001 = self._fetch_links_sigma(links001)
            sigma010 = self._fetch_links_sigma(links010)
            sigma011 = self._fetch_links_sigma(links011)
            sigma100 = self._fetch_links_sigma(links100)
            sigma101 = self._fetch_links_sigma(links101)
            sigma110 = self._fetch_links_sigma(links110)
            sigma111 = self._fetch_links_sigma(links111)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -self.opt.step_size
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )

            depth = weight * t * delta_scale[good_indices]

            out_depth[good_indices] = out_depth[good_indices] + depth
            log_light_intensity[good_indices] = log_light_intensity[good_indices] + log_att
            t = t + self.opt.step_size

            mask = t <= tmax
            good_indices = good_indices[mask].clone()
            origins = origins[mask].clone()
            dirs = dirs[mask].clone()
            t = t[mask]
            tmax = tmax[mask]

        return out_depth
    
    def volume_render(
        self, rays: Rays, use_kernel: bool = True,
        return_raylen: bool=False
    ):
        """
        Standard volume rendering. See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :param return_raylen: bool, if true then only returns the length of the
                                    ray-cube intersection and quits
        :return: (N, 3), predicted RGB
        """
        if use_kernel and self.links.is_cuda and _C is not None and not return_raylen:
            assert rays.is_cuda
            return _VolumeRenderFunction.apply(
                self.density_data,
                self.sh_data,
                self._to_cpp(),
                rays._to_cpp(),
                self.opt._to_cpp(),
            )
        else:
            warn("Using slow volume rendering, should only be used for debugging")
            return self._volume_render_rgbd_pytorch(rays)
                
    def volume_render_fused(
        self,
        rays: Rays,
        rgb_gt: torch.Tensor,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0
    ):
        """
        Standard volume rendering with fused MSE gradient generation,
            given a ground truth color for each pixel.
        Will update the *.grad tensors for each parameter
        You can then subtract the grad manually or use the optim_*_step methods

        See grid.opt.* (RenderOptions) for configs.

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param rgb_gt: (N, 3), GT pixel colors, each channel in [0, 1]
        :param beta_loss: float, weighting for beta loss to add to the gradient.
                                 (fused into the backward pass).
                                 This is average voer the rays in the batch.
                                 Beta loss also from neural volumes:
                                 [Lombardi et al., ToG 2019]
        :return: (N, 3), predicted RGB
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for fused"
        assert rays.is_cuda
        grad_density, grad_sh = self._get_data_grads()
        rgb_out = torch.zeros_like(rgb_gt)

        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),),
                dtype=torch.bool, device=self.density_data.device)

        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        grad_holder.mask_out = self.sparse_grad_indexer

        cu_fn = _C.__dict__[f"volume_render_cuvol_fused"]
        #  with utils.Timing("actual_render"):
        cu_fn(
            self._to_cpp(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            beta_loss,
            sparsity_loss,
            rgb_out,
            grad_holder
        )
        self.sparse_sh_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out
    
    def volume_render_fused_rgbd(
        self,
        rays: Rays,
        rgb_gt: torch.Tensor,
        depth_gt: torch.Tensor = None,
        beta_loss: float = 0.0,
        sparsity_loss: float = 0.0,
        d_loss: float = 0.0,
    ):
        assert rays.is_cuda
        grad_density, grad_sh = self._get_data_grads()
        self.sparse_grad_indexer = torch.zeros((self.density_data.size(0),), 
                dtype=torch.bool, device=self.density_data.device)
        
        grad_holder = _C.GridOutputGrads()
        grad_holder.grad_density_out = grad_density
        grad_holder.grad_sh_out = grad_sh
        grad_holder.mask_out = self.sparse_grad_indexer

        rgb_out = torch.zeros_like(rgb_gt)
        depth_out = torch.zeros_like(depth_gt)

        _C.volume_render_fused_rgbd(
            self._to_cpp(),
            rays._to_cpp(),
            self.opt._to_cpp(),
            rgb_gt,
            depth_gt,
            beta_loss,
            sparsity_loss,
            d_loss,
            rgb_out,
            depth_out,
            grad_holder)
    
        self.sparse_intensity_grad_indexer = self.sparse_grad_indexer.clone()
        return rgb_out, depth_out
    
    def volume_render_image(
        self, camera: Camera, use_kernel: bool = True,
        batch_size : int = 5000,
        return_raylen: bool=False
    ):
        """
        Standard volume rendering (entire image version).
        See grid.opt.* (RenderOptions) for configs.

        :param camera: Camera
        :param use_kernel: bool, if false uses pure PyTorch version even if on CUDA.
        :return: (H, W, 3), predicted RGB image
        """
        imrend_fn_name = f"volume_render_cuvol_image"
        if imrend_fn_name in _C.__dict__ and not torch.is_grad_enabled() and not return_raylen:
            # Use the fast image render kernel if available
            cu_fn = _C.__dict__[imrend_fn_name]
            return cu_fn(
                self._to_cpp(),
                camera._to_cpp(),
                self.opt._to_cpp()
            )
        else:
            # Manually generate rays for now
            rays = camera.gen_rays()
            all_rgb_out = []
            for batch_start in range(0, camera.height * camera.width, batch_size):
                rgb_out_part = self.volume_render(rays[batch_start:batch_start+batch_size],
                                                  use_kernel=use_kernel,
                                                  return_raylen=return_raylen)
                all_rgb_out.append(rgb_out_part)

            all_rgb_out = torch.cat(all_rgb_out, dim=0)
            return all_rgb_out.view(camera.height, camera.width, -1)

    def volume_render_pos_rgbd(self, rays):
        self.origins, self.dirs = rays.origins, rays.dirs

        return _RenderFunctionPoseRGBD.apply(
                self.origins,
                self.dirs,
                self._to_cpp(),
                self.opt._to_cpp())
    
    def volume_render_pos_rgb(self, rays):
        self.origins, self.dirs = rays.origins, rays.dirs

        return _RenderFunctionPoseRGB.apply(
                self.origins,
                self.dirs,
                self._to_cpp(),
                self.opt._to_cpp())

    def volume_render_depth(
            self, rays: Rays, sigma_thresh: Optional[float] = None,
            use_kernel: bool = True):
        """
        Volumetric depth rendering for rays

        :param rays: Rays, (origins (N, 3), dirs (N, 3))
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: (N,)
        """
        if use_kernel and self.links.is_cuda and _C is not None:
            if sigma_thresh is None:
                return _C.volume_render_expected_term(
                        self._to_cpp(),
                        rays._to_cpp(),
                        self.opt._to_cpp())
            else:
                return _C.volume_render_sigma_thresh(
                        self._to_cpp(),
                        rays._to_cpp(),
                        self.opt._to_cpp(),
                        sigma_thresh)
        else:
            return self._volume_render_depth_pytorch(rays)

    def volume_render_depth_image(self, camera: Camera, sigma_thresh: Optional[float] = None, batch_size: int = 5000):
        """
        Volumetric depth rendering for full image

        :param camera: Camera, a single camera
        :param sigma_thresh: Optional[float]. If None then finds the standard expected termination
                                              (NOTE: this is the absolute length along the ray, not the z-depth as usually expected);
                                              else then finds the first point where sigma strictly exceeds sigma_thresh

        :return: depth (H, W)
        """
        rays = camera.gen_rays()
        all_depths = []
        for batch_start in range(0, camera.height * camera.width, batch_size):
            depths = self.volume_render_depth(rays[batch_start: batch_start + batch_size], sigma_thresh)
            all_depths.append(depths)
        all_depth_out = torch.cat(all_depths, dim=0)
        return all_depth_out.view(camera.height, camera.width)

    def resample(
        self,
        reso: Union[int, List[int]],
        sigma_thresh: float = 5.0,
        weight_thresh: float = 0.01,
        dilate: int = 2,
        cameras: Optional[List[Camera]] = None,
        use_z_order: bool=False,
        accelerate: bool=True,
        weight_render_stop_thresh: float = 0.2, # SHOOT, forgot to turn this off for main exps..
        max_elements:int=0
    ):
        """
        Resample and sparsify the grid; used to increase the resolution
        :param reso: int or List[int, int, int], resolution for resampled grid, as in the constructor
        :param sigma_thresh: float, threshold to apply on the sigma (if using sigma thresh i.e. cameras NOT given)
        :param weight_thresh: float, threshold to apply on the weights (if using weight thresh i.e. cameras given)
        :param dilate: int, if true applies dilation of size <dilate> to the 3D mask for nodes to keep in the grid
                             (keep neighbors in all 28 directions, including diagonals, of the desired nodes)
        :param cameras: Optional[List[Camera]], optional list of cameras in OpenCV convention (if given, uses weight thresholding)
        :param use_z_order: bool, if true, stores the data initially in a Z-order curve if possible
        :param accelerate: bool, if true (default), calls grid.accelerate() after resampling
                           to build distance transform table (only if on CUDA)
        :param weight_render_stop_thresh: float, stopping threshold for grid weight render in [0, 1];
                                                 0.0 = no thresholding, 1.0 = hides everything.
                                                 Useful for force-cutting off
                                                 junk that contributes very little at the end of a ray
        :param max_elements: int, if nonzero, an upper bound on the number of elements in the
                upsampled grid; we will adjust the threshold to match it
        """
        with torch.no_grad():
            device = self.links.device
            if isinstance(reso, int):
                reso = [reso] * 3
            else:
                assert (
                    len(reso) == 3
                ), "reso must be an integer or indexable object of 3 ints"

            if use_z_order and not (reso[0] == reso[1] and reso[0] == reso[2] and utils.is_pow2(reso[0])):
                print("Morton code requires a cube grid of power-of-2 size, ignoring...")
                use_z_order = False

            self.capacity: int = reduce(lambda x, y: x * y, reso)
            curr_reso = self.links.shape
            dtype = torch.float32
            reso_facts = [0.5 * curr_reso[i] / reso[i] for i in range(3)]
            X = torch.linspace(
                reso_facts[0] - 0.5,
                curr_reso[0] - reso_facts[0] - 0.5,
                reso[0],
                dtype=dtype,
            )
            Y = torch.linspace(
                reso_facts[1] - 0.5,
                curr_reso[1] - reso_facts[1] - 0.5,
                reso[1],
                dtype=dtype,
            )
            Z = torch.linspace(
                reso_facts[2] - 0.5,
                curr_reso[2] - reso_facts[2] - 0.5,
                reso[2],
                dtype=dtype,
            )
            X, Y, Z = torch.meshgrid(X, Y, Z)
            points = torch.stack((X, Y, Z), dim=-1).view(-1, 3)

            if use_z_order:
                morton = utils.gen_morton(reso[0], dtype=torch.long).view(-1)
                points[morton] = points.clone()
            points = points.to(device=device)

            use_weight_thresh = cameras is not None

            batch_size = 720720
            all_sample_vals_density = []
            print('Pass 1/2 (density)')
            for i in tqdm(range(0, len(points), batch_size)):
                sample_vals_density, _ = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=False
                )
                sample_vals_density = sample_vals_density
                all_sample_vals_density.append(sample_vals_density)
            self.density_data.grad = None
            self.sh_data.grad = None
            self.sparse_grad_indexer = None
            self.sparse_sh_grad_indexer = None
            self.density_rms = None
            self.sh_rms = None

            sample_vals_density = torch.cat(
                    all_sample_vals_density, dim=0).view(reso)
            del all_sample_vals_density
            if use_weight_thresh:
                gsz = torch.tensor(reso)
                offset = (self._offset * gsz - 0.5).to(device=device)
                scaling = (self._scaling * gsz).to(device=device)
                max_wt_grid = torch.zeros(reso, dtype=torch.float32, device=device)
                print(" Grid weight render", sample_vals_density.shape)
                for i, cam in enumerate(cameras):
                    _C.grid_weight_render(
                        sample_vals_density, cam._to_cpp(),
                        0.5,
                        weight_render_stop_thresh,
                        False,
                        offset, scaling, max_wt_grid
                    )
                 
                sample_vals_mask = max_wt_grid >= weight_thresh
                if max_elements > 0 and max_elements < max_wt_grid.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    weight_thresh_bounded = torch.topk(max_wt_grid.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    weight_thresh = max(weight_thresh, weight_thresh_bounded)
                    print(' Readjusted weight thresh to fit to memory:', weight_thresh)
                    sample_vals_mask = max_wt_grid >= weight_thresh
                del max_wt_grid
            else:
                sample_vals_mask = sample_vals_density >= sigma_thresh
                if max_elements > 0 and max_elements < sample_vals_density.numel() \
                                    and max_elements < torch.count_nonzero(sample_vals_mask):
                    # To bound the memory usage
                    sigma_thresh_bounded = torch.topk(sample_vals_density.view(-1),
                                     k=max_elements, sorted=False).values.min().item()
                    sigma_thresh = max(sigma_thresh, sigma_thresh_bounded)
                    print(' Readjusted sigma thresh to fit to memory:', sigma_thresh)
                    sample_vals_mask = sample_vals_density >= sigma_thresh

                if self.opt.last_sample_opaque:
                    # Don't delete the last z layer
                    sample_vals_mask[:, :, -1] = 1

            if dilate:
                for i in range(int(dilate)):
                    sample_vals_mask = _C.dilate(sample_vals_mask)
            sample_vals_mask = sample_vals_mask.view(-1)
            sample_vals_density = sample_vals_density.view(-1)
            sample_vals_density = sample_vals_density[sample_vals_mask]
            cnz = torch.count_nonzero(sample_vals_mask).item()

            # Now we can get the colors for the sparse points
            points = points[sample_vals_mask]
            print('Pass 2/2 (color), eval', cnz, 'sparse pts')
            all_sample_vals_sh = []
            for i in tqdm(range(0, len(points), batch_size)):
                _, sample_vals_sh = self.sample(
                    points[i : i + batch_size],
                    grid_coords=True,
                    want_colors=True
                )
                all_sample_vals_sh.append(sample_vals_sh)

            sample_vals_sh = torch.cat(all_sample_vals_sh, dim=0) if len(all_sample_vals_sh) else torch.empty_like(self.sh_data[:0])
            del self.density_data
            del self.sh_data
            del all_sample_vals_sh

            if use_z_order:
                inv_morton = torch.empty_like(morton)
                inv_morton[morton] = torch.arange(morton.size(0), dtype=morton.dtype)
                inv_idx = inv_morton[sample_vals_mask]
                init_links = torch.full(
                    (sample_vals_mask.size(0),), fill_value=-1, dtype=torch.int32
                )
                init_links[inv_idx] = torch.arange(inv_idx.size(0), dtype=torch.int32)
            else:
                init_links = (
                    torch.cumsum(sample_vals_mask.to(torch.int32), dim=-1).int() - 1
                )
                init_links[~sample_vals_mask] = -1

            self.capacity = cnz
            print(" New cap:", self.capacity)
            del sample_vals_mask
            print('density', sample_vals_density.shape, sample_vals_density.dtype)
            print('sh', sample_vals_sh.shape, sample_vals_sh.dtype)
            print('links', init_links.shape, init_links.dtype)
            self.density_data = nn.Parameter(sample_vals_density.view(-1, 1).to(device=device))
            self.sh_data = nn.Parameter(sample_vals_sh.to(device=device))
            self.links = init_links.view(reso).to(device=device)

            if accelerate and self.links.is_cuda:
                self.accelerate()

    def resize(self, sh_dim: int):
        """
        Modify the size of the data stored in the voxels. Called expand/shrink in svox 1.

        :param sh_dim: new basis dimension, must be square number
        """
        assert utils.isqrt(sh_dim) is not None, "sh_dim (SH) must be a square number"
        assert (
            sh_dim >= 1 and sh_dim <= utils.MAX_SH_BASIS
        ), f"sh_dim 1-{utils.MAX_SH_BASIS} supported"
        old_sh_dim = self.sh_dim
        self.sh_dim = sh_dim
        device = self.sh_data.device
        old_data = self.sh_data.data.cpu()

        shrinking = sh_dim < old_sh_dim
        sigma_arr = torch.tensor([0])
        if shrinking:
            shift = old_sh_dim
            arr = torch.arange(sh_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])
        else:
            shift = sh_dim
            arr = torch.arange(old_sh_dim)
            remap = torch.cat([arr, shift + arr, 2 * shift + arr])

        del self.sh_data
        new_data = torch.zeros((old_data.size(0), 3 * sh_dim + 1), device="cpu")
        if shrinking:
            new_data[:] = old_data[..., remap]
        else:
            new_data[..., remap] = old_data
        new_data = new_data.to(device=device)
        self.sh_data = nn.Parameter(new_data)
        self.sh_rms = None

    def accelerate(self):
        """
        Accelerate
        """
        assert (
            _C is not None and self.links.is_cuda
        ), "CUDA extension is currently required for accelerate"
        _C.accel_dist_prop(self.links)

    def world2grid(self, points):
        """
        World coordinates to grid coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        offset = self._offset * gsz - 0.5
        scaling = self._scaling * gsz
        return torch.addcmul(
            offset.to(device=points.device), points, scaling.to(device=points.device)
        )

    def grid2world(self, points):
        """
        Grid coordinates to world coordinates. Grid coordinates are
        normalized to [0, n_voxels] in each side

        :param points: (N, 3)
        :return: (N, 3)
        """
        gsz = self._grid_size()
        roffset = self.radius * (1.0 / gsz - 1.0) + self.center
        rscaling = 2.0 * self.radius / gsz
        return torch.addcmul(
            roffset.to(device=points.device), points, rscaling.to(device=points.device)
        )

    def save(self, path: str, compress: bool = False):
        """
        Save to a path
        """
        save_fn = np.savez_compressed if compress else np.savez
        data = {
            "radius":self.radius.numpy(),
            "center":self.center.numpy(),
            "links":self.links.cpu().numpy(),
            "density_data":self.density_data.data.cpu().numpy(),
            "sh_data":self.sh_data.data.cpu().numpy().astype(np.float16),
        }
        save_fn(
            path,
            **data
        )

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "cpu"):
        """
        Load from path
        """
        z = np.load(path)
        if "data" in z.keys():
            # Compatibility
            all_data = z.f.data
            sh_data = all_data[..., 1:]
            density_data = all_data[..., :1]
        else:
            sh_data = z.f.sh_data
            density_data = z.f.density_data

        links = z.f.links
        sh_dim = (sh_data.shape[1]) // 3
        radius = z.f.radius.tolist() if "radius" in z.files else [1.0, 1.0, 1.0]
        center = z.f.center.tolist() if "center" in z.files else [0.0, 0.0, 0.0]
        grid = cls(
            1,
            radius=radius,
            center=center,
            sh_dim=sh_dim,
            use_z_order=False,
            device="cpu")
        if sh_data.dtype != np.float32:
            sh_data = sh_data.astype(np.float32)
        if density_data.dtype != np.float32:
            density_data = density_data.astype(np.float32)
        sh_data = torch.from_numpy(sh_data).to(device=device)
        density_data = torch.from_numpy(density_data).to(device=device)
        grid.sh_data = nn.Parameter(sh_data)
        grid.density_data = nn.Parameter(density_data)
        grid.links = torch.from_numpy(links).to(device=device)
        grid.capacity = grid.sh_data.size(0)
        
        if grid.links.is_cuda:
            grid.accelerate()
        return grid

    def tv(self, logalpha: bool=False, logalpha_delta: float=2.0):
        """
        Compute total variation over sigma,
        similar to Neural Volumes [Lombardi et al., ToG 2019]

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        return _TotalVariationFunction.apply(
                self.density_data, self.links, 0, 1, logalpha, logalpha_delta, False)

    def tv_color(self,
                 start_dim: int = 0, end_dim: Optional[int] = None,
                 logalpha: bool=False, logalpha_delta: float=2.0):
        """
        Compute total variation on color

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.

        :return: torch.Tensor, size scalar, the TV value (sum over channels,
                 mean over voxels)
        """
        assert (
            _C is not None and self.sh_data.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim
        return _TotalVariationFunction.apply(
            self.sh_data, self.links, start_dim, end_dim, logalpha, logalpha_delta, True)

    def inplace_tv_grad(self, grad: torch.Tensor,
                        scaling: float = 1.0,
                        sparse_frac: float = 0.01,
                        logalpha: bool=False, logalpha_delta: float=2.0,
                        contiguous: bool = True
                    ):
        """
        Add gradient of total variation for sigma as in Neural Volumes
        [Lombardi et al., ToG 2019]
        directly into the gradient tensor, multiplied by 'scaling'
        """
        assert (
            _C is not None and self.density_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"

        assert not logalpha, "No longer supported"
        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                _C.tv_grad_sparse(self.links, self.density_data,
                        rand_cells,
                        self._get_sparse_grad_indexer(),
                        0, 1, scaling,
                        logalpha, logalpha_delta,
                        False,
                        self.opt.last_sample_opaque,
                        grad)
        else:
            _C.tv_grad(self.links, self.density_data, 0, 1, scaling,
                    logalpha, logalpha_delta,
                    False,
                    grad)
            self.sparse_grad_indexer : Optional[torch.Tensor] = None

    def inplace_tv_color_grad(
        self,
        grad: torch.Tensor,
        start_dim: int = 0,
        end_dim: Optional[int] = None,
        scaling: float = 1.0,
        sparse_frac: float = 0.01,
        logalpha: bool=False,
        logalpha_delta: float=2.0,
        contiguous: bool = True
    ):
        """
        Add gradient of total variation for color
        directly into the gradient tensor, multiplied by 'scaling'

        :param start_dim: int, first color channel dimension to compute TV over (inclusive).
                          Default 0.
        :param end_dim: int, last color channel dimension to compute TV over (exclusive).
                          Default None = all dimensions until the end.
        """
        assert (
            _C is not None and self.sh_data.is_cuda and grad.is_cuda
        ), "CUDA extension is currently required for total variation"
        assert not logalpha, "No longer supported"
        if end_dim is None:
            end_dim = self.sh_data.size(1)
        end_dim = end_dim + self.sh_data.size(1) if end_dim < 0 else end_dim
        start_dim = start_dim + self.sh_data.size(1) if start_dim < 0 else start_dim

        rand_cells = self._get_rand_cells(sparse_frac, contiguous=contiguous)
        if rand_cells is not None:
            if rand_cells.size(0) > 0:
                indexer = self._get_sparse_sh_grad_indexer()
                #  with utils.Timing("actual_tv_color"):
                _C.tv_grad_sparse(self.links, self.sh_data,
                                  rand_cells,
                                  indexer,
                                  start_dim, end_dim, scaling,
                                  logalpha,
                                  logalpha_delta,
                                  True,
                                  False,
                                  grad)
        else:
            _C.tv_grad(self.links, self.sh_data, start_dim, end_dim, scaling,
                    logalpha,
                    logalpha_delta,
                    True,
                    grad)
            self.sparse_sh_grad_indexer = None

    def optim_density_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8):
        assert (_C is not None and self.sh_data.is_cuda), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer()
        if (self.density_rms is None
            or self.density_rms.shape != self.density_data.shape):
            del self.density_rms
            self.density_rms = torch.zeros_like(self.density_data.data) # FIXME init?
        _C.rmsprop_step(
            self.density_data.data,
            self.density_rms,
            self.density_data.grad,
            indexer,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def optim_sh_step(self, lr: float, beta: float=0.9, epsilon: float = 1e-8,
                      optim: str = 'rmsprop'):
        assert (_C is not None and self.sh_data.is_cuda), "CUDA extension is currently required for optimizers"

        indexer = self._maybe_convert_sparse_grad_indexer(sh=True)
        if self.sh_rms is None or self.sh_rms.shape != self.sh_data.shape:
            del self.sh_rms
            self.sh_rms = torch.zeros_like(self.sh_data.data) # FIXME init?
        _C.rmsprop_step(
            self.sh_data.data,
            self.sh_rms,
            self.sh_data.grad,
            indexer,
            beta,
            lr,
            epsilon,
            -1e9,
            lr
        )

    def __repr__(self):
        return (
            f"svox2.SparseGrid(sh_dim={self.sh_dim}, "
            + f"reso={list(self.links.shape)}, "
            + f"capacity:{self.sh_data.size(0)})"
        )

    def _to_cpp(self, grid_coords: bool = False):
        """
        Generate object to pass to C++
        """
        gspec = _C.SparseGridSpec()
        gspec.density_data = self.density_data
        gspec.sh_data = self.sh_data
        gspec.links = self.links
        if grid_coords:
            gspec._offset = torch.zeros_like(self._offset)
            gspec._scaling = torch.ones_like(self._offset)
        else:
            gsz = self._grid_size()
            gspec._offset = self._offset * gsz - 0.5
            gspec._scaling = self._scaling * gsz

        gspec.sh_dim = self.sh_dim

        return gspec

    def _grid_size(self):
        return torch.tensor(self.links.shape, device="cpu", dtype=torch.float32)

    def _get_data_grads(self):
        ret = []
        for subitem in ["density_data", "sh_data"]:
            param = self.__getattr__(subitem)
            if not param.requires_grad:
                ret.append(torch.zeros_like(param.data))
            else:
                if (
                    not hasattr(param, "grad")
                    or param.grad is None
                    or param.grad.shape != param.data.shape
                ):
                    if hasattr(param, "grad"):
                        del param.grad
                    param.grad = torch.zeros_like(param.data)
                ret.append(param.grad)
        return ret

    def _get_sparse_grad_indexer(self):
        indexer = self.sparse_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _get_sparse_sh_grad_indexer(self):
        indexer = self.sparse_sh_grad_indexer
        if indexer is None:
            indexer = torch.empty((0,), dtype=torch.bool, device=self.density_data.device)
        return indexer

    def _maybe_convert_sparse_grad_indexer(self, sh=False):
        """
        Automatically convert sparse grad indexer from mask to
        indices, if it is efficient
        """
        indexer = self.sparse_sh_grad_indexer if sh else self.sparse_grad_indexer

        if indexer is None:
            return torch.empty((), device=self.density_data.device)
        if (
            indexer.dtype == torch.bool and
            torch.count_nonzero(indexer).item()
            < indexer.size(0) // 8
        ):
            # Highly sparse (use index)
            indexer = torch.nonzero(indexer.flatten(), as_tuple=False).flatten()
        return indexer

    def _get_rand_cells(self, sparse_frac: float, force: bool = False, contiguous:bool=True):
        if sparse_frac < 1.0 or force:
            assert self.sparse_grad_indexer is None or self.sparse_grad_indexer.dtype == torch.bool, \
                   "please call sparse loss after rendering and before gradient updates"
            grid_size = self.links.size(0) * self.links.size(1) * self.links.size(2)
            sparse_num = max(int(sparse_frac * grid_size), 1)
            if contiguous:
                start = np.random.randint(0, grid_size)
                arr = torch.arange(start, start + sparse_num, dtype=torch.int32, device=
                                                self.links.device)

                if start > grid_size - sparse_num:
                    arr[grid_size - sparse_num - start:] -= grid_size
                return arr
            else:
                return torch.randint(0, grid_size, (sparse_num,), dtype=torch.int32, device=
                                                self.links.device)
        return None