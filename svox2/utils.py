from functools import partial
import torch
from torch import nn
from typing import Optional, Tuple
import numpy as np
from dataclasses import dataclass
import math

def inthroot(x : int, n : int):
    if x <= 0:
        return None
    lo, hi = 1, x
    while lo <= hi:
        mi = lo + (hi - lo) // 2
        p = mi ** n
        if p == x:
            return mi
        elif p > x:
            hi = mi - 1
        else:
            lo = mi + 1
    return None

isqrt = partial(inthroot, n=2)

def is_pow2(x : int):
    return x > 0 and (x & (x - 1)) == 0

def _get_c_extension():
    from warnings import warn
    try:
        import svox2.csrc as _C
        if not hasattr(_C, "sample_grid"):
            _C = None
    except:
        _C = None

    if _C is None:
        warn("CUDA extension svox2.csrc could not be loaded! " +
             "Operations will be slow.\n" +
             "Please do not import svox in the svox2 source directory.")
    return _C

# Morton code (Z-order curve)
def _expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v

def _unexpand_bits(v):
    v &= 0x49249249
    v = (v | (v >> 2)) & 0xc30c30c3
    v = (v | (v >> 4)) & 0xf00f00f
    v = (v | (v >> 8)) & 0xff0000ff
    v = (v | (v >> 16)) & 0x0000ffff
    return v


def morton_code_3(x, y, z):
    xx = _expand_bits(x)
    yy = _expand_bits(y)
    zz = _expand_bits(z)
    return (xx << 2) + (yy << 1) + zz

def inv_morton_code_3(code):
    x = _unexpand_bits(code >> 2)
    y = _unexpand_bits(code >> 1)
    z = _unexpand_bits(code)
    return x, y, z

def gen_morton(D, device='cpu', dtype=torch.long):
    assert is_pow2(D), "Morton code requires power of 2 reso"
    arr = torch.arange(D, device=device, dtype=dtype)
    X, Y, Z = torch.meshgrid(arr, arr, arr)
    mort = morton_code_3(X, Y, Z)
    return mort


# SH

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10
def eval_sh_bases(sh_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.

    :param sh_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions

    :return: torch.Tensor (..., sh_dim)
    """
    result = torch.empty((*dirs.shape[:-1], sh_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if sh_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if sh_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if sh_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if sh_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

# Ray-sphere intersector for MSI
class ConcentricSpheresIntersector:
    def __init__(self,
            size : torch.Tensor,
            rorigins : torch.Tensor,
            rdirs : torch.Tensor,
            rworld_step : torch.Tensor):
        sphere_scaling = 2.0 / size

        origins = (rorigins + 0.5) * sphere_scaling - 1.0
        dirs = rdirs * sphere_scaling
        inorm = 1.0 / dirs.norm(dim=-1)

        self.world_step_scale = rworld_step * inorm
        dirs *= inorm.unsqueeze(-1)

        self.q2a : torch.Tensor = 2 * (dirs * dirs).sum(-1)
        self.qb : torch.Tensor = 2 * (origins * dirs).sum(-1)
        self.f = self.qb.square() - 2 * self.q2a * (origins * origins).sum(-1)
        self.origins = origins
        self.dirs = dirs

    def intersect(self, r : float):
        """
        Find far intersection of all rays with sphere of radius r
        """
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] +
                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def intersect_near(self, r : float):
        """
        Find near intersection of all rays with sphere of radius r
        """
        det = self._det(r)
        success_mask = det >= 0
        result = torch.zeros_like(self.q2a)
        result[success_mask] = (-self.qb[success_mask] -
                torch.sqrt(det[success_mask])) / self.q2a[success_mask]
        return success_mask, result

    def _det(self, r : float) -> torch.Tensor:
        return self.f + 2 * self.q2a * (r * r)


def memlog(device='cuda'):
    # Memory debugging
    print(torch.cuda.memory_summary(device))
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                    hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if str(obj.device) != 'cpu':
                    print(obj.device, '{: 10}'.format(obj.numel()),
                            obj.dtype,
                            obj.size(), type(obj))
        except:
            pass


def spher2cart(theta : torch.Tensor, phi : torch.Tensor):
    """Convert spherical coordinates into Cartesian coordinates on unit sphere."""
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

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

