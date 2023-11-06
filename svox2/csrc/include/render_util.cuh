// Copyright 2021 Alex Yu
#pragma once

#include <cstdint>
#include "data_spec_packed.cuh"
#include "random_util.cuh"

namespace {
namespace device {

template<class data_type_t, class voxel_index_t>
__device__ __inline__ float trilerp_one(
        const data_type_t* __restrict__ data,
        int reso, int stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const int offz = stride;
    const int offy = reso * stride;
    const int offx = reso * offy;
    const data_type_t* __restrict__ data_ptr = data + (offx * l[0] +
                                                    offy * l[1] +
                                                    offz * l[2]
                                                    + idx);

    const float ix0y0 = lerp(data_ptr[0], data_ptr[offz], pos[2]);
    const float ix0y1 = lerp(data_ptr[offy], data_ptr[offy + offz], pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(data_ptr[offx], data_ptr[offx + offz], pos[2]);
    const float ix1y1 = lerp(data_ptr[offy + offx],
                             data_ptr[offy + offx + offz], pos[2]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_one(
        data_type_t* __restrict__ grad_data,
        int reso, int stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int offz = stride;
    const int offy = reso * stride;
    const int offx = reso * offy;
    data_type_t* __restrict__ grad_data_ptr = grad_data + (offx * l[0] +
                                                    offy * l[1] +
                                                    offz * l[2]
                                                    + idx);

#define ADD_WT(u, val) atomicAdd(&grad_data_ptr[u], val)
    ADD_WT(0, ay * az * xo);
    ADD_WT(offz, ay * pos[2] * xo);
    ADD_WT(offy, pos[1] * az * xo);
    ADD_WT(offy + offz, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    ADD_WT(offx, ay * az * xo);
    ADD_WT(offx + offz, ay * pos[2] * xo);
    ADD_WT(offx + offy, pos[1] * az * xo);
    ADD_WT(offx + offy + offz, pos[1] * pos[2] * xo);
#undef ADD_WT
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ float put_color(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx,
        float color) {
    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define PUTCOLOR(u, val) if (link_ptr[u] >= 0) { data[link_ptr[u] * stride + idx]=val;}

    PUTCOLOR(0, color);
    PUTCOLOR(1, color);
    PUTCOLOR(offy, color);
    PUTCOLOR(offy + 1, color);
    PUTCOLOR(offx, color);
    PUTCOLOR(offx + 1, color);
    PUTCOLOR(offy + offx, color);
    PUTCOLOR(offy + offx+1, color);
#undef PUTCOLOR
}

// trilerp with links
template<class data_type_t, class voxel_index_t>
__device__ __inline__ float trilerp_cuvol_one(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? data[link_ptr[u] * stride + idx] : 0.f)
    const float ix0y0 = lerp(MAYBE_READ_LINK(0), MAYBE_READ_LINK(1), pos[2]);
    const float ix0y1 = lerp(MAYBE_READ_LINK(offy), MAYBE_READ_LINK(offy + 1), pos[2]);
    const float ix0 = lerp(ix0y0, ix0y1, pos[1]);
    const float ix1y0 = lerp(MAYBE_READ_LINK(offx), MAYBE_READ_LINK(offx + 1), pos[2]);
    const float ix1y1 = lerp(MAYBE_READ_LINK(offy + offx),
                             MAYBE_READ_LINK(offy + offx + 1), pos[2]);
    const float ix1 = lerp(ix1y0, ix1y1, pos[1]);
    return lerp(ix0, ix1, pos[0]);
#undef MAYBE_READ_LINK
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_cuvol_one(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data,
        int offx, int offy, size_t stride,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out,
        const int idx) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data[link_ptr[u] * stride + idx], val); \
        }
    MAYBE_ADD_LINK(0, ay * az * xo);
    MAYBE_ADD_LINK(1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offy + 1, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    MAYBE_ADD_LINK(offx + 0, ay * az * xo);
    MAYBE_ADD_LINK(offx + 1, ay * pos[2] * xo);
    MAYBE_ADD_LINK(offx + offy, pos[1] * az * xo);
    MAYBE_ADD_LINK(offx + offy + 1, pos[1] * pos[2] * xo);
#undef MAYBE_ADD_LINK
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_cuvol_one_density(
        const int32_t* __restrict__ links,
        data_type_t* __restrict__ grad_data_out,
        bool* __restrict__ mask_out,
        int offx, int offy,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        float grad_out) {
    const float ay = 1.f - pos[1], az = 1.f - pos[2];
    float xo = (1.0f - pos[0]) * grad_out;

    const int32_t* __restrict__ link_ptr = links + (offx * l[0] + offy * l[1] + l[2]);

#define MAYBE_ADD_LINK_DEN(u, val) if (link_ptr[u] >= 0) { \
              atomicAdd(&grad_data_out[link_ptr[u]], val); \
              if (mask_out != nullptr) \
                  mask_out[link_ptr[u]] = true; \
        }
    MAYBE_ADD_LINK_DEN(0, ay * az * xo);
    MAYBE_ADD_LINK_DEN(1, ay * pos[2] * xo);
    MAYBE_ADD_LINK_DEN(offy, pos[1] * az * xo);
    MAYBE_ADD_LINK_DEN(offy + 1, pos[1] * pos[2] * xo);

    xo = pos[0] * grad_out;
    MAYBE_ADD_LINK_DEN(offx + 0, ay * az * xo);
    MAYBE_ADD_LINK_DEN(offx + 1, ay * pos[2] * xo);
    MAYBE_ADD_LINK_DEN(offx + offy, pos[1] * az * xo);
    MAYBE_ADD_LINK_DEN(offx + offy + 1, pos[1] * pos[2] * xo);
#undef MAYBE_ADD_LINK_DEN
}

// START POSE OPTIM UTILS
__device__ __inline__ void get_trilerp_const(
    const float* __restrict__ c_data,
    float* __restrict__ a_data) {
    a_data[0] = c_data[0];
    a_data[1] = -c_data[0] + c_data[4];
    a_data[2] = -c_data[0] + c_data[2];
    a_data[3] = -c_data[0] + c_data[1];
    a_data[4] = c_data[0] - c_data[2] - c_data[4] + c_data[6];
    a_data[5] = c_data[0] - c_data[1] - c_data[4] + c_data[5];
    a_data[6] = c_data[0] - c_data[1] - c_data[2] + c_data[3];
    a_data[7] = -c_data[0] + c_data[1] + c_data[2] - c_data[3] \
                + c_data[4] - c_data[5] - c_data[6] + c_data[7];
}

__device__ __inline__ void trilerp_deriv_pos(
    const float* __restrict__ a_data,
    const float* __restrict__ pos,
    float* __restrict__ deriv){
    deriv[0] = a_data[1] + a_data[4]*pos[1] + a_data[5]*pos[2] + a_data[7]*pos[1]*pos[2];
    deriv[1] = a_data[2] + a_data[4]*pos[0] + a_data[6]*pos[2] + a_data[7]*pos[0]*pos[2];
    deriv[2] = a_data[3] + a_data[5]*pos[0] + a_data[6]*pos[1] + a_data[7]*pos[0]*pos[1];
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_pos_color(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ sh_data,
        data_type_t* __restrict__ grad_trilerp_color, //out
        int reso, 
        size_t sh_data_dim,
        const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos,
        const int idx) {
    const int offz = 1;
    const int offy = reso*offz;
    const int offx = reso*offy;

    float color_voxel_val[8];
    const int32_t* __restrict__ link_ptr = links + (offx*l[0] + offy*l[1] + l[2]);

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? sh_data[link_ptr[u]*sh_data_dim + idx] : 0.f)
    color_voxel_val[0] = MAYBE_READ_LINK(0);           // C000 = 0
    color_voxel_val[1] = MAYBE_READ_LINK(1);           // C001 = 1
    color_voxel_val[2] = MAYBE_READ_LINK(offy);        // C010 = offy
    color_voxel_val[3] = MAYBE_READ_LINK(offy+1);      // C011 = offy+1
    color_voxel_val[4] = MAYBE_READ_LINK(offx);        // C100 = offx
    color_voxel_val[5] = MAYBE_READ_LINK(offx+1);      // C101 = offx+1
    color_voxel_val[6] = MAYBE_READ_LINK(offx+offy);   // C110 = offx+offy
    color_voxel_val[7] = MAYBE_READ_LINK(offx+offy+1); // C111 = offx+offy+1
#undef MAYBE_READ_LINK

    float trilin_color_const[8];

    get_trilerp_const(
        color_voxel_val,
        trilin_color_const
    );
    trilerp_deriv_pos(
        trilin_color_const, 
        pos, 
        grad_trilerp_color //output
    );
}

template<class data_type_t, class voxel_index_t>
__device__ __inline__ void trilerp_backward_pos_density(
        const int32_t* __restrict__ links,
        const data_type_t* __restrict__ density_data,
        data_type_t* __restrict__ grad_trilerp_density, //out
        int reso, const voxel_index_t* __restrict__ l,
        const float* __restrict__ pos) {

    const int offz = 1;
    const int offy = reso*offz;
    const int offx = reso*offy;

    float density_voxel_val[8];

    const int32_t* __restrict__ link_ptr = links + (offx*l[0] + offy*l[1] + l[2]);

#define MAYBE_READ_LINK(u) ((link_ptr[u] >= 0) ? density_data[link_ptr[u]] : 0.f)
    density_voxel_val[0] = MAYBE_READ_LINK(0);           // C000 = 0
    density_voxel_val[1] = MAYBE_READ_LINK(1);           // C001 = 1
    density_voxel_val[2] = MAYBE_READ_LINK(offy);        // C010 = offy
    density_voxel_val[3] = MAYBE_READ_LINK(offy+1);      // C011 = offy+1
    density_voxel_val[4] = MAYBE_READ_LINK(offx);        // C100 = offx
    density_voxel_val[5] = MAYBE_READ_LINK(offx+1);      // C101 = offx+1
    density_voxel_val[6] = MAYBE_READ_LINK(offx+offy);   // C110 = offx+offy
    density_voxel_val[7] = MAYBE_READ_LINK(offx+offy+1); // C111 = offx+offy+1
#undef MAYBE_READ_LINK

    float trilin_density_const[8];

    get_trilerp_const(
        density_voxel_val,
        trilin_density_const
    );

    trilerp_deriv_pos(
        trilin_density_const, 
        pos, 
        grad_trilerp_density //output
    );
}

// Compute the amount to skip for negative link values
__device__ __inline__ float compute_skip_dist(
        SingleRaySpec& __restrict__ ray,
        const int32_t* __restrict__ links,
        int offx, int offy,
        int pos_offset = 0) {
    const int32_t link_val = links[offx * (ray.l[0] + pos_offset) +
                                   offy * (ray.l[1] + pos_offset) +
                                   (ray.l[2] + pos_offset)];
    if (link_val >= -1) return 0.f; // Not worth

    const uint32_t dist = -link_val;
    const uint32_t cell_ul_shift = (dist - 1);
    const uint32_t cell_side_len = (1 << cell_ul_shift) - 1.f;

    // AABB intersection
    // Consider caching the invdir for the ray
    float tmin = 0.f;
    float tmax = 1e9f;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        int ul = (((ray.l[i] + pos_offset) >> cell_ul_shift) << cell_ul_shift);
        ul -= ray.l[i] + pos_offset;

        const float invdir = 1.0 / ray.dir[i];
        const float t1 = (ul - ray.pos[i] + pos_offset) * invdir;
        const float t2 = (ul + cell_side_len - ray.pos[i] + pos_offset) * invdir;
        if (ray.dir[i] != 0.f) {
            tmin = max(tmin, min(t1, t2));
            tmax = min(tmax, max(t1, t2));
        }
    }

//     const uint32_t cell_ul_shift = 1 - dist;
//     const uint32_t cell_br_shift = -cell_ul_shift;
//
//     // AABB intersection
//     // Consider caching the invdir for the ray
//     float tmin = 0.f;
//     float tmax = 1e9f;
// #pragma unroll
//     for (int i = 0; i < 3; ++i) {
//         const float invdir = 1.0 / ray.dir[i];
//         const float t1 = (cell_ul_shift - ray.pos[i] + pos_offset) * invdir;
//         const float t2 = (cell_br_shift - ray.pos[i] + pos_offset) * invdir;
//         if (ray.dir[i] != 0.f) {
//             tmin = max(tmin, min(t1, t2));
//             tmax = min(tmax, max(t1, t2));
//         }
//     }

    if (tmin > 0.f) {
        // Somehow the origin is not in the cube
        // Should not happen for distance transform

        // If using geometric distances:
        // will happen near the lowest vertex of a cell,
        // since l is always the lowest neighbor
        return 0.f;
    }
    return tmax;
}

// Spherical functions

// SH Coefficients from https://github.com/google/spherical-harmonics
__device__ __constant__ const float C0 = 0.28209479177387814;
__device__ __constant__ const float C1 = 0.4886025119029199;
__device__ __constant__ const float C2[] = {
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
};

__device__ __constant__ const float C3[] = {
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
};

// __device__ __constant__ const float C4[] = {
//     2.5033429417967046,
//     -1.7701307697799304,
//     0.9461746957575601,
//     -0.6690465435572892,
//     0.10578554691520431,
//     -0.6690465435572892,
//     0.47308734787878004,
//     -1.7701307697799304,
//     0.6258357354491761,
// };

__device__ __inline__ void calc_sh(
    const int sh_dim,
    const float* __restrict__ dir,
    float* __restrict__ out) {
    out[0] = C0;
    const float x = dir[0], y = dir[1], z = dir[2];
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, yz = y * z, xz = x * z;
    switch (sh_dim) {
        // 16 not supported rn due to warp size
        // case 16:
        //     out[9] = C3[0] * y * (3 * xx - yy);
        //     out[10] = C3[1] * xy * z;
        //     out[11] = C3[2] * y * (4 * zz - xx - yy);
        //     out[12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
        //     out[13] = C3[4] * x * (4 * zz - xx - yy);
        //     out[14] = C3[5] * z * (xx - yy);
        //     out[15] = C3[6] * x * (xx - 3 * yy);
        //     [[fallthrough]];
        case 9:
            out[4] = C2[0] * xy;
            out[5] = C2[1] * yz;
            out[6] = C2[2] * (2.0 * zz - xx - yy);
            out[7] = C2[3] * xz;
            out[8] = C2[4] * (xx - yy);
            [[fallthrough]];
        case 4:
            out[1] = -C1 * y;
            out[2] = C1 * z;
            out[3] = -C1 * x;
    }
}

__device__ __inline__ void calc_sphfunc(
    const PackedSparseGridSpec& grid,
    const int lane_id,
    const int ray_id,
    const float* __restrict__ dir, // Pre-normalized
    float* __restrict__ out) {
    calc_sh(grid.sh_dim, dir, out);
}

__device__ __inline__ float _intersect_aabb_unit(
        const float* __restrict__ cen,
        const float* __restrict__ invdir) {
    // Intersect unit AABB
    float tmax = 1e9f;
    float t1, t2;
#pragma unroll
    for (int i = 0; i < 3; ++i) {
        t1 = - cen[i] * invdir[i];
        t2 = t1 +  invdir[i];
        tmax = min(tmax, max(t1, t2));
    }
    return tmax;
}

__device__ __inline__ float _get_delta_scale(
    const float* __restrict__ scaling,
    float* __restrict__ dir) {
    dir[0] *= scaling[0];
    dir[1] *= scaling[1];
    dir[2] *= scaling[2];
    float delta_scale = _rnorm(dir);
    dir[0] *= delta_scale;
    dir[1] *= delta_scale;
    dir[2] *= delta_scale;
    return delta_scale;
}

__device__ __inline__ static void _normalize(
                float* __restrict__ dir) {
    const float rnorm = _rnorm(dir);
    dir[0] *= rnorm; dir[1] *= rnorm; dir[2] *= rnorm;
}

__device__ __inline__ static void _unitvec2equirect(
        const float* __restrict__ unit_dir,
        int reso,
        float* __restrict__ xy) {
    const float lat = asinf(unit_dir[1]);
    const float lon = atan2f(unit_dir[0], unit_dir[2]);
    xy[0] = reso * 2 * (0.5 + lon * 0.5 * M_1_PI);
    xy[1] = reso * (0.5 - lat * M_1_PI);
}

__device__ __inline__ static void _equirect2unitvec(
        float x, float y,
        int reso,
        float* __restrict__ unit_dir) {
    const float lon = (x * (1.0 / (reso * 2)) - 0.5) * (2 * M_PI);
    const float lat = -(y * (1.0 / reso) - 0.5) * M_PI;
    const float coslat = cosf(lat);
    unit_dir[0] = coslat * sinf(lon);
    unit_dir[1] = sinf(lat);
    unit_dir[2] = coslat * cosf(lon);
}

__device__ __inline__ void cam2world_ray(
    int ix, int iy,
    const PackedCameraSpec& __restrict__ cam,
    // Outputs
    float* __restrict__ dir,
    float* __restrict__ origin) {
    // OpenCV convention (contrary to svox 1, which uses OpenGL)
    float x = (ix + 0.5f - cam.cx) / cam.fx;
    float y = (iy + 0.5f - cam.cy) / cam.fy;
    float z = sqrtf(x * x + y * y + 1.0);
    x /= z; y /= z; z = 1.0f / z;
    dir[0] = cam.c2w[0][0] * x + cam.c2w[0][1] * y + cam.c2w[0][2] * z;
    dir[1] = cam.c2w[1][0] * x + cam.c2w[1][1] * y + cam.c2w[1][2] * z;
    dir[2] = cam.c2w[2][0] * x + cam.c2w[2][1] * y + cam.c2w[2][2] * z;
    origin[0] = cam.c2w[0][3]; origin[1] = cam.c2w[1][3]; origin[2] = cam.c2w[2][3];
}

struct ConcentricSpheresIntersector {
    __device__
        ConcentricSpheresIntersector(
                const float* __restrict__ origin,
                const float* __restrict__ dir)
    {
        q2a = 2 * _dot(dir, dir);
        qb = 2 * _dot(origin, dir);
        f = qb * qb - 2 * q2a * _dot(origin, origin);
    }

    // Get the far intersection, which we want for rendering MSI
    __device__
    bool intersect(float r, float* __restrict__ out, bool near=false) {
        float det = _det(r);
        if (det < 0) return false;
        if (near) {
            *out = (-qb - sqrtf(det)) / q2a;
        } else {
            *out = (-qb + sqrtf(det)) / q2a;
        }
        return true;
    }

    __device__ __host__
    float _det (float r) {
        return f + 2 * q2a * r * r;
    }

    float q2a, qb, f;
};

__device__ __inline__ void ray_find_bounds(
        SingleRaySpec& __restrict__ ray,
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt,
        uint32_t ray_id) {
    // Warning: modifies ray.origin
    transform_coord(ray.origin, grid._scaling, grid._offset);
    
    // Warning: modifies ray.dir
    ray.world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;
    if (opt.use_spheric_clip) {
        const float sphere_scaling[3] {
            2.f / float(grid.size[0]),
            2.f / float(grid.size[1]),
            2.f / float(grid.size[2])
        };
        float sph_origin[3], sph_dir[3];

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            sph_origin[i] = fmaf(ray.origin[i] + 0.5f, sphere_scaling[i], -1.f);
            sph_dir[i] = ray.dir[i] * sphere_scaling[i];
        }

        ConcentricSpheresIntersector csi(sph_origin, sph_dir);
        if (!csi.intersect(1.f, &ray.tmax) || !csi.intersect(1.f - opt.near_clip, &ray.tmin, true)) {
            ray.tmin = 1e-9f;
            ray.tmax = 0.f;
        }
    } else {
        ray.tmin = opt.near_clip / ray.world_step * opt.step_size;
        ray.tmax = 2e3f;
        for (int i = 0; i < 3; ++i) {
            const float invdir = 1.0 / ray.dir[i];
            const float t1 = (-0.5f - ray.origin[i]) * invdir;
            const float t2 = (grid.size[i] - 0.5f  - ray.origin[i]) * invdir;
            if (ray.dir[i] != 0.f) {
                ray.tmin = max(ray.tmin, min(t1, t2));
                ray.tmax = min(ray.tmax, max(t1, t2));
            }
        }
    }
}

__device__ __inline__ float ray_find_bounds_rscale(
        SingleRaySpec& __restrict__ ray,
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt) {
    transform_coord(ray.origin, grid._scaling, grid._offset);
    float delta_scale;
    delta_scale = _get_delta_scale(grid._scaling, ray.dir);
    ray.world_step = delta_scale*opt.step_size;
    ray.tmin = opt.near_clip / ray.world_step*opt.step_size;
    ray.tmax = 2e3f;
    for (int i = 0; i < 3; ++i) {
        const float invdir = 1.0 / ray.dir[i];
        const float t1 = (-0.5f - ray.origin[i])*invdir;
        const float t2 = (grid.size[i] - 0.5f  - ray.origin[i])*invdir;
        if (ray.dir[i] != 0.f) {
            ray.tmin = max(ray.tmin, min(t1, t2));
            ray.tmax = min(ray.tmax, max(t1, t2));
        }
    }
    return delta_scale;
}
} // namespace device
} // namespace
