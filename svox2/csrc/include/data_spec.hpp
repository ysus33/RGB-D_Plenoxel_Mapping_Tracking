// Copyright 2021 Alex Yu
#pragma once
#include "util.hpp"
#include <torch/extension.h>

using torch::Tensor;

struct SparseGridSpec {
  Tensor density_data;
  Tensor sh_data;
  Tensor links;
  Tensor _offset;
  Tensor _scaling;
  int sh_dim;

  inline void check() {
    CHECK_INPUT(density_data);
    CHECK_INPUT(sh_data);
    CHECK_INPUT(links);
    
    CHECK_CPU_INPUT(_offset);
    CHECK_CPU_INPUT(_scaling);
    TORCH_CHECK(density_data.ndimension() == 2);
    TORCH_CHECK(sh_data.ndimension() == 2);
    TORCH_CHECK(links.ndimension() == 3);
  }
};

struct GridOutputGrads {
  torch::Tensor grad_density_out;
  torch::Tensor grad_sh_out;
  torch::Tensor mask_out;
  inline void check() {
    if (grad_density_out.defined()) {
      CHECK_INPUT(grad_density_out);
    }
    if (grad_sh_out.defined()) {
      CHECK_INPUT(grad_sh_out);
    }
    if (mask_out.defined() && mask_out.size(0) > 0) {
      CHECK_INPUT(mask_out);
    }
  }
};

struct CameraSpec {
  torch::Tensor c2w;
  float fx;
  float fy;
  float cx;
  float cy;
  int width;
  int height;

  inline void check() {
    CHECK_INPUT(c2w);
    TORCH_CHECK(c2w.is_floating_point());
    TORCH_CHECK(c2w.ndimension() == 2);
    TORCH_CHECK(c2w.size(1) == 4);
  }
};

struct RaysSpec {
  Tensor origins;
  Tensor dirs;
  inline void check() {
    CHECK_INPUT(origins);
    CHECK_INPUT(dirs);
    TORCH_CHECK(origins.is_floating_point());
    TORCH_CHECK(dirs.is_floating_point());
  }
};

struct RenderOptions {
  float empty_space_brightness;
  float step_size;
  float sigma_thresh;
  float stop_thresh;
  float near_clip;
  bool use_spheric_clip;
  bool last_sample_opaque;
};


struct PoseGrads {
  Tensor grad_pose_origin_out;
  Tensor grad_pose_direction_out;
  inline void check() {
    if (grad_pose_origin_out.defined()) {
      CHECK_INPUT(grad_pose_origin_out);
    }
    if (grad_pose_direction_out.defined()) {
      CHECK_INPUT(grad_pose_direction_out);
    }
  }
};