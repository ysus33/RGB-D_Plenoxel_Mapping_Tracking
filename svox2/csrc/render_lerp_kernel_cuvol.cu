// Copyright 2021 Alex Yu
#include <torch/extension.h>
#include "cuda_util.cuh"
#include "data_spec_packed.cuh"
#include "render_util.cuh"

#include <iostream>
#include <cstdint>
#include <tuple>

namespace {
const int WARP_SIZE = 32;

const int TRACE_RAY_CUDA_THREADS = 128;
const int TRACE_RAY_CUDA_RAYS_PER_BLOCK = TRACE_RAY_CUDA_THREADS / WARP_SIZE;

// const int TRACE_RAY_BKWD_CUDA_THREADS = 128;
const int TRACE_RAY_BKWD_CUDA_THREADS = 1024;

const int TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK = TRACE_RAY_BKWD_CUDA_THREADS / WARP_SIZE;

const int MIN_BLOCKS_PER_SM = 8;

typedef cub::WarpReduce<float> WarpReducef;

namespace device {


__device__ __inline__ void render_ys(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = opt.empty_space_brightness;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);

        // link_ptr = links + (grid.stride_x * ray.l[0] + grid.size[2] * ray.l[1] + ray.l[2]);

        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }

        if (sigma > 1) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            // if (_EXP(log_transmit) < opt.stop_thresh) {
                // log_transmit = -1e3f;
                break;
            // }
        }
        t += opt.step_size;
    }

    outv += _EXP(log_transmit) * opt.empty_space_brightness;
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        out[lane_colorgrp] = outv;
        while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);

        // link_ptr = links + (grid.stride_x * ray.l[0] + grid.size[2] * ray.l[1] + ray.l[2]);

        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }


        if (sigma > opt.sigma_thresh) {
            put_color(
                grid.links, out,
                grid.stride_x,
                grid.size[2],
                3,
                ray.l, ray.pos,
                lane_colorgrp,
                outv
            );
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        
        }
        t += opt.step_size;
    }


    }
}

// * For ray rendering
__device__ __inline__ void trace_ray_cuvol(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;

    if (ray.tmin > ray.tmax) {
        out[lane_colorgrp] = opt.empty_space_brightness;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }

        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    outv += _EXP(log_transmit) * opt.empty_space_brightness;
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        out[lane_colorgrp] = outv;
    }
}

__device__ __inline__ void trace_ray_cuvol_d_new(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ rgb_out,
        float* __restrict__ d_out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;

    if (ray.tmin > ray.tmax) {
        rgb_out[lane_colorgrp] = opt.empty_space_brightness;
        *d_out = 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    float rgb_outv = 0.f;
    float d_outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }

        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;


            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);
            rgb_outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            d_outv += weight * (t / opt.step_size) * ray.world_step;
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }

    rgb_outv += _EXP(log_transmit) * opt.empty_space_brightness;
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr) {
            *out_log_transmit = log_transmit;
        }
        rgb_out[lane_colorgrp] = rgb_outv;
        *d_out = d_outv;
    }
}

__device__ __inline__ void trace_ray_expected_term(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    float t = ray.tmin;
    float outv = 0.f;

    float log_transmit = 0.f;
    // printf("tmin %f, tmax %f \n", ray.tmin, ray.tmax);

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > opt.sigma_thresh) {
            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            outv += weight * (t / opt.step_size) * ray.world_step;
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    *out = outv;
}

// From Dex-NeRF
__device__ __inline__ void trace_ray_sigma_thresh(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        float sigma_thresh,
        float* __restrict__ out) {
    if (ray.tmin > ray.tmax) {
        *out = 0.f;
        return;
    }

    float t = ray.tmin;
    *out = 0.f;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > sigma_thresh) {
            *out = (t / opt.step_size) * ray.world_step;
            break;
        }
        t += opt.step_size;
    }
}

__device__ __inline__ void trace_ray_cuvol_backward(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,
        float* __restrict__ grad_sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        PackedGridOutputGrads& __restrict__ grads,
        float* __restrict__ accum_out,
        float* __restrict__ log_transmit_out
        ) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;
    const uint32_t leader_mask = 1U | (1U << grid.sh_dim) | (1U << (2 * grid.sh_dim));

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    float t = ray.tmin;

    const float gout = grad_output[lane_colorgrp];

    float log_transmit = 0.f;

    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.sh_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.sh_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.sh_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (
                    total_color * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));
            }
            trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one_density(
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) { *log_transmit_out = log_transmit; }
    }
}

__device__ __inline__ void trace_ray_cuvol_backward_d_new(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_out_rgb,
        float grad_out_d,
        const float* __restrict__ rgb_cache,
        float d_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,
        float* __restrict__ grad_sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        float d_loss,
        PackedGridOutputGrads& __restrict__ grads,
        float* __restrict__ accum_out,
        float* __restrict__ log_transmit_out
        ) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;
    const uint32_t leader_mask = 1U | (1U << grid.sh_dim) | (1U << (2 * grid.sh_dim));

    float accum = fmaf(rgb_cache[0], grad_out_rgb[0],
                      fmaf(rgb_cache[1], grad_out_rgb[1],
                           rgb_cache[2] * grad_out_rgb[2])) + d_loss * d_cache * grad_out_d;

    if (ray.tmin > ray.tmax) {
        if (accum_out != nullptr) { *accum_out = accum; }
        if (log_transmit_out != nullptr) { *log_transmit_out = 0.f; }
        // printf("accum_end_fg_fast=%f\n", accum);
        return;
    }

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3)); // d beta_loss / d log_transmit_in
        accum += beta_loss;
        // Interesting how this loss turns out, kinda nice?
    }

    float t = ray.tmin;

    const float gout = grad_out_rgb[lane_colorgrp];

    float log_transmit = 0.f;

    // remat samples
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }
        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            float depth = (t/opt.step_size) * ray.world_step;

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_depth = depth * grad_out_d;

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.sh_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.sh_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.sh_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;

            accum -= weight*(total_color + d_loss*total_depth);

            float curr_grad_sigma = ray.world_step*((total_color + d_loss*total_depth) * _EXP(log_transmit) - accum);
            if (sparsity_loss > 0.f) {
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));
            }
            trilerp_backward_cuvol_one(grid.links, grads.grad_sh_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0) {
                trilerp_backward_cuvol_one_density(
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(log_transmit) < opt.stop_thresh) {
                break;
            }
        }
        t += opt.step_size;
    }
    if (lane_id == 0) {
        if (accum_out != nullptr) {
            // Cancel beta loss out in case of background
            accum -= beta_loss;
            *accum_out = accum;
        }
        if (log_transmit_out != nullptr) { *log_transmit_out = log_transmit; }
        // printf("accum_end_fg=%f\n", accum);
        // printf("log_transmit_fg=%f\n", log_transmit);
    }
}

__device__ __inline__ void trace_ray_pos_backward(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_output,
        const float* __restrict__ color_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ grads_ori,
        float* __restrict__ grads_dir) // 현재 ray에서의 계산결과가 저장될 곳의 포인터. [x,y,x]
{
    float grads_ori_[3] = {0., 0., 0.,};
    float grads_dir_[3] = {0., 0., 0.,};


    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;
    const uint32_t leader_mask = 1U | (1U << grid.sh_dim) | (1U << (2 * grid.sh_dim));
    // no idea what this is for

    float accum = fmaf(color_cache[0], grad_output[0],
                      fmaf(color_cache[1], grad_output[1],
                           color_cache[2] * grad_output[2]));

    if (ray.tmin > ray.tmax) {
        return;
    }
    
    float t = ray.tmin;
    const float gout = grad_output[lane_colorgrp];
    float log_transmit = 0.f;
    int tot_len = (ray.tmax-ray.tmin)/opt.step_size;

    if (tot_len <= 0){
        // printf("ray, out of bounds, return \n");
        return;
    }
    
    int stepcount = 0;
    while (t <= ray.tmax) {
        #pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;

            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout;  // Clamp to [+0, infty)

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.sh_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.sh_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.sh_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;
            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (total_color * _EXP(log_transmit) - accum); //inside here there is gout (= dL(C)/dC_hat)

            float grad_trilerp_color[3]; // [dσi/dxi, dσi/dyi, dσi/dzi]

            // what is the best way to get a color_voxel_val(rendered with full sh, 
            // single color(one of R, G, B)) in here?
            trilerp_backward_pos_color(
                    grid.links, 
                    grid.sh_data, 
                    grad_trilerp_color, // (dci/dxi, dci/dyi, dci/dzi)
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos, lane_id);
            // printf("%d) curr_grad_sigma: %f\n", stepcount, curr_grad_sigma);
            // dCi/dOxi
            float grad_origin_xi = curr_grad_color * grad_trilerp_color[0];
            float grad_origin_yi = curr_grad_color * grad_trilerp_color[1];
            float grad_origin_zi = curr_grad_color * grad_trilerp_color[2];

            if (lane_id == 0) {
                float grad_trilerp_density[3]; // [dσi/dxi, dσi/dyi, dσi/dzi]
                trilerp_backward_pos_density(
                    grid.links, 
                    grid.density_data,
                    grad_trilerp_density, // (dσi/dxi, dσi/dyi, dσi/dzi) //this is computed 9 times!
                    grid.size[2],
                    ray.l, ray.pos);
                grad_origin_xi += curr_grad_sigma * grad_trilerp_density[0];
                grad_origin_yi += curr_grad_sigma * grad_trilerp_density[1];
                grad_origin_zi += curr_grad_sigma * grad_trilerp_density[2];
            }
            // P = O + tD
            // dP/dO = 1 -> correct (pos.register_hook(), origins.register_hook())
            // dP/dD = t -> I'm not sure
            grads_ori_[0] += grad_origin_xi; // dCi/dPxi = (dCi/dci * dci/dPxi) + (dCi/dσi * dσi/dPxi)
            grads_ori_[1] += grad_origin_yi;
            grads_ori_[2] += grad_origin_zi;

            // float depth = (t/opt.step_size) * ray.world_step;
            // printf("t: %f, step_size: %f, ray.world_step: %f, depth: %f\n", t, opt.step_size, ray.world_step, depth);
            grads_dir_[0] += t * grad_origin_xi; // dCi/dDxi
            grads_dir_[1] += t * grad_origin_yi; // dCi/dDyi
            grads_dir_[2] += t * grad_origin_zi; // dCi/dDzi
        } 
        // printf("lane_colorgrp: %d, lane_colorgrp_id: %d, stepcount: %d, t: %f, curr_grad_color: %f, lane_color_total: %f\n", lane_colorgrp, lane_colorgrp_id, stepcount, t, curr_grad_color, lane_color_total);

        t += opt.step_size;
        stepcount++;
    }
        // printf("%d th partial origin grad: [%f, %f, %f]\n", lane_id,
        //     grads_ori_[0], 
        //     grads_ori_[1], 
        //     grads_ori_[2]);

        // printf("%d th partial dir grad: [%f, %f, %f]\n", lane_id,
        //     grads_dir_[0], 
        //     grads_dir_[1], 
        //     grads_dir_[2]);

    // printf("%d th partial origin grad: [%f, %f, %f]\n", lane_id,
    //         grads_ori_[0] * grid._scaling[0], 
    //         grads_ori_[1] * grid._scaling[1], 
    //         grads_ori_[2] * grid._scaling[2]);

    atomicAdd(grads_ori, grads_ori_[0]* grid._scaling[0]);
    atomicAdd(grads_ori+1, grads_ori_[1]* grid._scaling[0]);
    atomicAdd(grads_ori+2, grads_ori_[2]* grid._scaling[0]);


    atomicAdd(grads_dir, grads_dir_[0]);
    atomicAdd(grads_dir+1, grads_dir_[1]);
    atomicAdd(grads_dir+2, grads_dir_[2]);
}

__device__ __inline__ void trace_ray_pos_backward_d(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ rgb_grad_out,
        float d_grad_out,
        const float* __restrict__ rgb_cache,
        float depth_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ grads_ori,
        float* __restrict__ grads_dir,
        float delta_scale) // 현재 ray에서의 계산결과가 저장될 곳의 포인터. [x,y,x]
{
    float grads_ori_[3] = {0., 0., 0.,};
    float grads_dir_[3] = {0., 0., 0.,};


    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;
    const uint32_t leader_mask = 1U | (1U << grid.sh_dim) | (1U << (2 * grid.sh_dim));
    // no idea what this is for

    float accum = fmaf(rgb_cache[0], rgb_grad_out[0],
                      fmaf(rgb_cache[1], rgb_grad_out[1],
                           rgb_cache[2] * rgb_grad_out[2]));


    if (ray.tmin > ray.tmax) {
        return;
    }
    
    float t = ray.tmin;
    const float gout = rgb_grad_out[lane_colorgrp];
    float gradout_d = d_grad_out;
    float accum_d = depth_cache * gradout_d;

    float log_transmit = 0.f;
    int tot_len = (ray.tmax-ray.tmin)/opt.step_size;
    float real_factor = delta_scale / grid._scaling[0];

    if (tot_len <= 0){
        return;
    }
    
    while (t <= ray.tmax) {
        #pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);
        if (skip >= opt.step_size) {
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links,
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links,
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];


            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;

            float total_color = fmaxf(lane_color_total, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout;  // Clamp to [+0, infty)

            
            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.sh_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.sh_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.sh_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;
            
            accum -= weight * total_color;
            float curr_grad_sigma = ray.world_step * (total_color * _EXP(log_transmit) - accum); //inside here there is gout (= dL(C)/dC_hat)
            
            float grad_trilerp_color[3]; // [dσi/dxi, dσi/dyi, dσi/dzi]

            trilerp_backward_pos_color(
                    grid.links, 
                    grid.sh_data, 
                    grad_trilerp_color, // (dci/dxi, dci/dyi, dci/dzi)
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos, lane_id);
            // printf("%d) curr_grad_sigma: %f\n", stepcount, curr_grad_sigma);
            // dCi/dOxi
            float grad_origin_xi = curr_grad_color * grad_trilerp_color[0];
            float grad_origin_yi = curr_grad_color * grad_trilerp_color[1];
            float grad_origin_zi = curr_grad_color * grad_trilerp_color[2];

            if (lane_id == 0) {
                float grad_trilerp_density[3]; // [dσi/dxi, dσi/dyi, dσi/dzi]
                trilerp_backward_pos_density(
                    grid.links, 
                    grid.density_data,
                    grad_trilerp_density, // (dσi/dxi, dσi/dyi, dσi/dzi) //this is computed 9 times!
                    grid.size[2],
                    ray.l, ray.pos);

                float depth = (t/opt.step_size) * ray.world_step;
                float total_depth = depth * gradout_d;
                accum_d -= weight * total_depth;
                float curr_grad_sigma_d = ray.world_step*(total_depth*_EXP(log_transmit) 
                                                        - accum_d);  // grad_c * (dCi/dσi) + grad_d * w_d * (dDi/dσi)

                float grad_dir_scal = weight*gradout_d;
                grad_origin_xi += (curr_grad_sigma+curr_grad_sigma_d) * grad_trilerp_density[0] \
                                + grad_dir_scal*ray.dir[0]*real_factor*real_factor;
                grad_origin_yi += (curr_grad_sigma+curr_grad_sigma_d) * grad_trilerp_density[1] \
                                + grad_dir_scal*ray.dir[1]*real_factor*real_factor;
                grad_origin_zi += (curr_grad_sigma+curr_grad_sigma_d) * grad_trilerp_density[2] \
                                + grad_dir_scal*ray.dir[2]*real_factor*real_factor;
            }
            grads_ori_[0] += grad_origin_xi;
            grads_ori_[1] += grad_origin_yi;
            grads_ori_[2] += grad_origin_zi;

            grads_dir_[0] += t * grad_origin_xi; // dCi/dDxi
            grads_dir_[1] += t * grad_origin_yi; // dCi/dDyi
            grads_dir_[2] += t * grad_origin_zi; // dCi/dDzi
        } 

        t += opt.step_size;
    }
    atomicAdd(grads_ori, grads_ori_[0]* grid._scaling[0]);
    atomicAdd(grads_ori+1, grads_ori_[1]* grid._scaling[0]);
    atomicAdd(grads_ori+2, grads_ori_[2]* grid._scaling[0]);

    atomicAdd(grads_dir, grads_dir_[0]);
    atomicAdd(grads_dir+1, grads_dir_[1]);
    atomicAdd(grads_dir+2, grads_dir_[2]);
}

// BEGIN KERNELS

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out[ray_id].data(),
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel_d_new(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rgb_out,
        float* __restrict__ depth_out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol_d_new(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        rgb_out[ray_id].data(),
        depth_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}


__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_image_kernel(
        PackedSparseGridSpec grid,
        PackedCameraSpec cam,
        RenderOptions opt,
        float* __restrict__ out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, cam.height * cam.width * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    const int ix = ray_id % cam.width;
    const int iy = ray_id / cam.width;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];

    cam2world_ray(ix, iy, cam, ray_spec[ray_blk_id].dir, ray_spec[ray_blk_id].origin);
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        out + ray_id * 3,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };
    if (lane_id < grid.sh_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }

    float grad_out[3];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float resid = color_cache[ray_id * 3 + i] - grad_output[ray_id * 3 + i];
            grad_out[i] = resid * norm_factor;
        }
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            grad_out[i] = grad_output[ray_id * 3 + i];
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_cuvol_backward(
        grid,
        grad_out,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel_d_new(
    PackedSparseGridSpec grid,
    const float* __restrict__ rgb_gt,
    const float* __restrict__ depth_gt,
    const float* __restrict__ rgb_cache,
    const float* __restrict__ depth_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    bool grad_out_is_rgb,
    const float* __restrict__ log_transmit_in,
    float beta_loss,
    float sparsity_loss,
    float d_loss,
    PackedGridOutputGrads grads,
    float* __restrict__ accum_out = nullptr,
    float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };
    if (lane_id < grid.sh_dim) {
        grad_sphfunc_val[ray_blk_id][lane_id] = 0.f;
    }
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 vdir, sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }

    float rgb_grad_out[3];
    float d_grad_out;

    const float d_resid = depth_cache[ray_id] - depth_gt[ray_id];
    if (grad_out_is_rgb) {
        const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            const float rgb_resid = rgb_cache[ray_id * 3 + i] - rgb_gt[ray_id * 3 + i];
            rgb_grad_out[i] = rgb_resid * norm_factor;
        }
        d_grad_out = d_resid * norm_factor;
    } else {
#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            rgb_grad_out[i] = rgb_gt[ray_id * 3 + i];
        }
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_cuvol_backward_d_new(
        grid,
        rgb_grad_out,
        d_grad_out,
        rgb_cache + ray_id * 3,
        //depth_cache + ray_id,
        depth_cache[ray_id],
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        d_loss,
        grads,
        accum_out == nullptr ? nullptr : accum_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_expected_term_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_expected_term(
        grid,
        ray_spec,
        opt,
        out + ray_id);
}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_sigma_thresh_kernel(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        float sigma_thresh,
        float* __restrict__ out) {
    CUDA_GET_THREAD_ID(ray_id, rays.origins.size(0));
    SingleRaySpec ray_spec(rays.origins[ray_id].data(), rays.dirs[ray_id].data());
    ray_find_bounds(ray_spec, grid, opt, ray_id);
    trace_ray_sigma_thresh(
        grid,
        ray_spec,
        opt,
        sigma_thresh,
        out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_pos_kernel(
    PackedSparseGridSpec grid,
    const float* __restrict__ grad_output,
    const float* __restrict__ color_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    PackedPoseGrads grads) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_pos_backward(
        grid,
        grad_output + ray_id * 3,
        color_cache + ray_id * 3,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        grads.grad_pose_origin_out + ray_id * 3,     // [x,y,z]
        grads.grad_pose_direction_out + ray_id * 3); // [x,y,z]
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_pos_kernel_d(
    PackedSparseGridSpec grid,
    const float* __restrict__ rgb_grad_out,
    const float* __restrict__ d_grad_out,
    const float* __restrict__ rgb_cache,
    const float* __restrict__ depth_cache,
    PackedRaysSpec rays,
    RenderOptions opt,
    PackedPoseGrads grads) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        // ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }
    float delta_scale;
    delta_scale = ray_find_bounds_rscale(ray_spec[ray_blk_id], grid, opt);

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_pos_backward_d(
        grid,
        rgb_grad_out + ray_id * 3,
        d_grad_out[ray_id],
        rgb_cache + ray_id * 3,
        depth_cache[ray_id],
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        grads.grad_pose_origin_out + ray_id * 3,     // [x,y,z]
        grads.grad_pose_direction_out + ray_id * 3,
        delta_scale); // [x,y,z]
}

__device__ __inline__ void trace_ray_cuvol_d(
        const PackedSparseGridSpec& __restrict__ grid,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        float* __restrict__ sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float* __restrict__ rgb_out,
        float* __restrict__ d_out,
        float* __restrict__ out_log_transmit) {
    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;

    if (ray.tmin > ray.tmax) {
        rgb_out[lane_colorgrp] = 0.f;
        *d_out = 0.f;
        if (out_log_transmit != nullptr) {
            *out_log_transmit = 0.f;
        }
        return;
    }

    float t = ray.tmin;
    
    float rgb_outv = 0.f;
    float d_outv = 0.f;

    float log_transmit = 0.f;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray, 
                        grid.links, grid.stride_x, 
                        grid.size[2], 0);
        
        if (skip >= opt.step_size) {
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links, 
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            lane_color *= sphfunc_val[lane_colorgrp_id]; // bank conflict

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           lane_color, lane_colorgrp_id == 0);

            rgb_outv += weight * fmaxf(lane_color_total + 0.5f, 0.f);  // Clamp to [+0, infty)
            d_outv += weight * (t / opt.step_size) * ray.world_step; //original, same as expected term in original plenoxel code
            
            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
            //just for temporal test
    // *d_out = (ray.tmax / opt.step_size) * ray.world_step;
    if (lane_colorgrp_id == 0) {
        if (out_log_transmit != nullptr){
            *out_log_transmit = log_transmit;
        }
        rgb_out[lane_colorgrp] = rgb_outv;
        *d_out = d_outv;
    }
}

__device__ __inline__ void trace_ray_backward_d(
        const PackedSparseGridSpec& __restrict__ grid,
        const float* __restrict__ grad_out_rgb,
        float grad_out_d,
        const float* __restrict__ rgb_cache,
        float depth_cache,
        SingleRaySpec& __restrict__ ray,
        const RenderOptions& __restrict__ opt,
        uint32_t lane_id,
        const float* __restrict__ sphfunc_val,
        float* __restrict__ grad_sphfunc_val,
        WarpReducef::TempStorage& __restrict__ temp_storage,
        float log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        float d_loss,
        PackedGridOutputGrads& __restrict__ grads) {

    const uint32_t lane_colorgrp_id = lane_id % grid.sh_dim;
    const uint32_t lane_colorgrp = lane_id / grid.sh_dim;
    const uint32_t leader_mask = 1U | (1U << grid.sh_dim) | (1U << (2 * grid.sh_dim));

    // float accum = rgb_cache * grad_out_rgb + d_loss * depth_cache * gout_d;

    float accum = fmaf(rgb_cache[0], grad_out_rgb[0],
                      fmaf(rgb_cache[1], grad_out_rgb[1],
                           rgb_cache[2] * grad_out_rgb[2])) + d_loss * depth_cache * grad_out_d;

    if (beta_loss > 0.f) {
        const float transmit_in = _EXP(log_transmit_in);
        beta_loss *= (1 - transmit_in / (1 - transmit_in + 1e-3));
        accum += beta_loss;
    }

    if (ray.tmin > ray.tmax) 
        return;

    float t = ray.tmin;
    float log_transmit = 0.f;
    const float gout = grad_out_rgb[lane_colorgrp];
    // float std = depth_cache * (1 - 0.03)/2;
    float std = 0.07;
    float deviation_loss = 1e-8;
    
    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray, grid.links, grid.stride_x, grid.size[2], 0);
        if (skip >= opt.step_size) {
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }

        float sigma = trilerp_cuvol_one(
                grid.links, 
                grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0); 
        if (sigma > opt.sigma_thresh) {
            float lane_color = trilerp_cuvol_one(
                            grid.links, 
                            grid.sh_data,
                            grid.stride_x,
                            grid.size[2],
                            grid.sh_data_dim,
                            ray.l, ray.pos, lane_id);
            float weighted_lane_color = lane_color * sphfunc_val[lane_colorgrp_id];

            float depth = (t/opt.step_size) * ray.world_step;

            const float pcnt = ray.world_step * sigma;
            const float weight = _EXP(log_transmit) * (1.f - _EXP(-pcnt));
            log_transmit -= pcnt;

            const float lane_color_total = WarpReducef(temp_storage).HeadSegmentedSum(
                                           weighted_lane_color, lane_colorgrp_id == 0) + 0.5f;
            float total_color = fmaxf(lane_color_total + 0.5f, 0.f);
            float color_in_01 = total_color == lane_color_total;
            total_color *= gout; // Clamp to [+0, infty)

            float total_depth = depth * grad_out_d;

            float total_color_c1 = __shfl_sync(leader_mask, total_color, grid.sh_dim);
            total_color += __shfl_sync(leader_mask, total_color, 2 * grid.sh_dim);
            total_color += total_color_c1;

            color_in_01 = __shfl_sync((1U << grid.sh_data_dim) - 1, color_in_01, lane_colorgrp * grid.sh_dim);
            const float grad_common = weight * color_in_01 * gout;
            const float curr_grad_color = sphfunc_val[lane_colorgrp_id] * grad_common;
            
            accum -= weight*(total_color + d_loss*total_depth);

            float curr_grad_sigma = ray.world_step*((total_color + d_loss*total_depth) * _EXP(log_transmit) - accum);
            if (deviation_loss > 0.f)
                // curr_grad_sigma += deviation_loss * 
                //         (1 - _EXP(-(depth - depth_cache)*(depth - depth_cache)/(2*std*std)));
                curr_grad_sigma += deviation_loss * (1 - _SIGMOID((depth-depth_cache*0.9)/(std*std)));

            if (sparsity_loss > 0.f)
                curr_grad_sigma += sparsity_loss * (4 * sigma / (1 + 2 * (sigma * sigma)));
            trilerp_backward_cuvol_one(
                    grid.links, 
                    grads.grad_sh_out,
                    grid.stride_x,
                    grid.size[2],
                    grid.sh_data_dim,
                    ray.l, ray.pos,
                    curr_grad_color, lane_id);
            if (lane_id == 0){
                trilerp_backward_cuvol_one_density(
                        grid.links,
                        grads.grad_density_out,
                        grads.mask_out,
                        grid.stride_x,
                        grid.size[2],
                        ray.l, ray.pos, curr_grad_sigma);
            }
            if (_EXP(log_transmit) < opt.stop_thresh)
                break;
        }
        t += opt.step_size;
    }

}

__launch_bounds__(TRACE_RAY_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_kernel_d(
        PackedSparseGridSpec grid,
        PackedRaysSpec rays,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rgb_out,
        float* __restrict__ depth_out,
        float* __restrict__ log_transmit_out = nullptr) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)  // Bad, but currently the best way due to coalesced memory access
        return;

    __shared__ float sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];

    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
            rays.dirs[ray_id].data());
    calc_sphfunc(grid, lane_id,
                 ray_id,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    __syncwarp((1U << grid.sh_data_dim) - 1);

    trace_ray_cuvol_d(
        grid,
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        rgb_out[ray_id].data(),
        depth_out + ray_id,
        log_transmit_out == nullptr ? nullptr : log_transmit_out + ray_id);
}

__launch_bounds__(TRACE_RAY_BKWD_CUDA_THREADS, MIN_BLOCKS_PER_SM)
__global__ void render_ray_backward_kernel_d(
        PackedSparseGridSpec grid,
        const float* __restrict__ rgb_gt,
        const float* __restrict__ depth_gt,
        const float* __restrict__ rgb_cache,
        const float* __restrict__ depth_cache,
        PackedRaysSpec rays,
        RenderOptions opt,
        bool grad_out_is_rgb,
        const float* __restrict__ log_transmit_in,
        float beta_loss,
        float sparsity_loss,
        float d_loss,
        PackedGridOutputGrads grads) {
    CUDA_GET_THREAD_ID(tid, int(rays.origins.size(0)) * WARP_SIZE);
    const int ray_id = tid >> 5;
    const int ray_blk_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 0x1F;

    if (lane_id >= grid.sh_data_dim)
        return;
        
    // printf("ray_id: {%d}, ray_blk_id: {%d}, lane_id: {%d} \n",
    // ray_id, ray_blk_id, lane_id);

    __shared__ float sphfunc_val[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK][9];
    __shared__ float grad_sphfunc_val[TRACE_RAY_CUDA_RAYS_PER_BLOCK][9];
    __shared__ SingleRaySpec ray_spec[TRACE_RAY_BKWD_CUDA_RAYS_PER_BLOCK];
    __shared__ typename WarpReducef::TempStorage temp_storage[
        TRACE_RAY_CUDA_RAYS_PER_BLOCK];

    ray_spec[ray_blk_id].set(rays.origins[ray_id].data(),
                             rays.dirs[ray_id].data());
    const float vdir[3] = {ray_spec[ray_blk_id].dir[0],
                     ray_spec[ray_blk_id].dir[1],
                     ray_spec[ray_blk_id].dir[2] };

    calc_sphfunc(grid, lane_id,
                 ray_id,
                //  vdir,
                 ray_spec[ray_blk_id].dir,
                 sphfunc_val[ray_blk_id]);
    if (lane_id == 0) {
        ray_find_bounds(ray_spec[ray_blk_id], grid, opt, ray_id);
    }
    float rgb_grad_out[3];
    float d_grad_out;
    const float norm_factor = 2.f / (3 * int(rays.origins.size(0)));
#pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        const float rgb_resid = rgb_cache[ray_id * 3 + i] - rgb_gt[ray_id * 3 + i];
        rgb_grad_out[i] = rgb_resid * norm_factor;
    }

    const float d_resid = depth_cache[ray_id] - depth_gt[ray_id];
    d_grad_out = d_resid * norm_factor;

    __syncwarp((1U << grid.sh_data_dim) - 1);
    trace_ray_backward_d(
        grid, 
        rgb_grad_out,
        d_grad_out,
        rgb_cache + ray_id * 3,
        depth_cache[ray_id],
        ray_spec[ray_blk_id],
        opt,
        lane_id,
        sphfunc_val[ray_blk_id],
        grad_sphfunc_val[ray_blk_id],
        temp_storage[ray_blk_id],
        log_transmit_in == nullptr ? 0.f : log_transmit_in[ray_id],
        beta_loss,
        sparsity_loss,
        d_loss,
        grads);
}
}  // namespace device

torch::Tensor _get_empty_1d(const torch::Tensor& origins) {
    auto options =
        torch::TensorOptions()
        .dtype(origins.dtype())
        .layout(torch::kStrided)
        .device(origins.device())
        .requires_grad(false);
    return torch::empty({origins.size(0)}, options);
}

}  // namespace

torch::Tensor volume_render_cuvol(SparseGridSpec& grid, RaysSpec& rays, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();


    const auto Q = rays.origins.size(0);

    torch::Tensor results = torch::empty_like(rays.origins);
    torch::Tensor log_transmit;

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_kernel<<<blocks, cuda_n_threads>>>(
                grid, rays, opt,
                // Output
                results.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                nullptr);
    }

    CUDA_CHECK_ERRORS;
    return results;
}

torch::Tensor volume_render_cuvol_image(SparseGridSpec& grid, CameraSpec& cam, RenderOptions& opt) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    cam.check();


    const auto Q = cam.height * cam.width;
    auto options =
        torch::TensorOptions()
        .dtype(grid.sh_data.dtype())
        .layout(torch::kStrided)
        .device(grid.sh_data.device())
        .requires_grad(false);

    torch::Tensor results = torch::empty({cam.height, cam.width, 3}, options);
    torch::Tensor log_transmit;

    {
        const int cuda_n_threads = TRACE_RAY_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads);
        device::render_ray_image_kernel<<<blocks, cuda_n_threads>>>(
                grid,
                cam,
                opt,
                // Output
                results.data_ptr<float>(),
                nullptr);
    }

    CUDA_CHECK_ERRORS;
    return results;
}

void volume_render_cuvol_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    torch::Tensor log_transmit, accum;
    
    {
        const int cuda_n_threads_render_backward = TRACE_RAY_BKWD_CUDA_THREADS;
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, cuda_n_threads_render_backward);
        device::render_ray_backward_kernel<<<blocks,
            cuda_n_threads_render_backward>>>(
                    grid,
                    grad_out.data_ptr<float>(),
                    color_cache.data_ptr<float>(),
                    rays, opt,
                    false, 
                    nullptr,
                    0.f,
                    0.f,
                    // Output
                    grads,
                    nullptr,
                    nullptr);
    }

    CUDA_CHECK_ERRORS;
}

void volume_render_cuvol_fused(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        float beta_loss,
        float sparsity_loss,
        torch::Tensor rgb_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    bool need_log_transmit = beta_loss > 0.f;
    torch::Tensor log_transmit, accum;
    if (need_log_transmit) {
        log_transmit = _get_empty_1d(rays.origins);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                // Output
                grads,
                nullptr,
                nullptr);
    }

    CUDA_CHECK_ERRORS;
}

torch::Tensor volume_render_expected_term(SparseGridSpec& grid,
        RaysSpec& rays, RenderOptions& opt) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_expected_term_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            results.data_ptr<float>()
        );
    return results;
}

torch::Tensor volume_render_sigma_thresh(SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        float sigma_thresh) {
    auto options =
        torch::TensorOptions()
        .dtype(rays.origins.dtype())
        .layout(torch::kStrided)
        .device(rays.origins.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({rays.origins.size(0)}, options);
    const auto Q = rays.origins.size(0);
    const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_CUDA_THREADS);
    device::render_ray_sigma_thresh_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
            grid,
            rays,
            opt,
            sigma_thresh,
            results.data_ptr<float>()
        );
    return results;
}

Tensor render_pos_forward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor color_out) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);
    torch::Tensor log_transmit;
    log_transmit = _get_empty_1d(rays.origins);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                    grid,
                    rays, 
                    opt,
                    // Output
                    color_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    log_transmit.data_ptr<float>());
    }
    CUDA_CHECK_ERRORS;
    return log_transmit;
}

Tensor render_pos_forward_rgbd(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_out,
        torch::Tensor depth_out) {
    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(depth_out);
    grid.check();
    rays.check();
    const auto Q = rays.origins.size(0);
    torch::Tensor log_transmit;
    log_transmit = _get_empty_1d(rays.origins);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel_d_new<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                    grid, rays, opt,
                    // Output
                    rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    depth_out.data_ptr<float>(),
                    log_transmit.data_ptr<float>());
    }
    CUDA_CHECK_ERRORS;
    return log_transmit;
}

void render_pos_backward(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor grad_out,
        torch::Tensor color_cache,
        PoseGrads& grads) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);
    // printf("new");

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_pos_kernel<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
        // device::render_ray_backward_pos_kernel<<<blocks*TRACE_RAY_BKWD_CUDA_THREADS,1>>>(
                    grid,
                    grad_out.data_ptr<float>(),
                    color_cache.data_ptr<float>(),
                    rays, 
                    opt,
                    // Output
                    grads);
    }
    CUDA_CHECK_ERRORS;
}

void render_pos_backward_rgbd(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_grad_out,
        torch::Tensor d_grad_out,
        torch::Tensor rgb_cache,
        torch::Tensor depth_cache,
        PoseGrads& grads) {
    DEVICE_GUARD(grid.sh_data);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_pos_kernel_d<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                    grid,
                    rgb_grad_out.data_ptr<float>(),
                    d_grad_out.data_ptr<float>(),
                    rgb_cache.data_ptr<float>(),
                    depth_cache.data_ptr<float>(),
                    rays, 
                    opt,
                    // Output
                    grads);
    }
    CUDA_CHECK_ERRORS;
}

void volume_render_fused_rgbd(
        SparseGridSpec& grid,
        RaysSpec& rays,
        RenderOptions& opt,
        torch::Tensor rgb_gt,
        torch::Tensor depth_gt,
        float beta_loss,
        float sparsity_loss,
        float d_loss,
        //output
        torch::Tensor rgb_out,
        torch::Tensor depth_out,
        GridOutputGrads& grads) {

    DEVICE_GUARD(grid.sh_data);
    CHECK_INPUT(rgb_gt);
    CHECK_INPUT(rgb_out);
    CHECK_INPUT(depth_gt);
    CHECK_INPUT(depth_out);
    grid.check();
    rays.check();
    grads.check();
    const auto Q = rays.origins.size(0);
    
    bool need_log_transmit = beta_loss > 0.f;
    torch::Tensor log_transmit;
    if (need_log_transmit) 
        log_transmit = _get_empty_1d(rays.origins);

    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q * WARP_SIZE, TRACE_RAY_CUDA_THREADS);
        device::render_ray_kernel_d_new<<<blocks, TRACE_RAY_CUDA_THREADS>>>(
                grid, rays, opt,
                // Output
                rgb_out.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                depth_out.data_ptr<float>(),
                need_log_transmit ? log_transmit.data_ptr<float>() : nullptr);
    }
    {
        const int blocks = CUDA_N_BLOCKS_NEEDED(Q, TRACE_RAY_BKWD_CUDA_THREADS);
        device::render_ray_backward_kernel_d_new<<<blocks, TRACE_RAY_BKWD_CUDA_THREADS>>>(
                grid,
                rgb_gt.data_ptr<float>(),
                depth_gt.data_ptr<float>(),
                rgb_out.data_ptr<float>(),
                depth_out.data_ptr<float>(),
                rays, opt,
                true,
                beta_loss > 0.f ? log_transmit.data_ptr<float>() : nullptr,
                beta_loss / Q,
                sparsity_loss,
                d_loss,
                // Output
                grads,
                nullptr, nullptr);
    }
    CUDA_CHECK_ERRORS;
}