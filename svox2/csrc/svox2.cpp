// Copyright 2021 Alex Yu
// This file contains only Python bindings
#include "data_spec.hpp"
#include <cstdint>
#include <torch/extension.h>
#include <tuple>

using torch::Tensor;

std::tuple<torch::Tensor, torch::Tensor> sample_grid(SparseGridSpec &, Tensor, bool);
void sample_grid_backward(SparseGridSpec &, Tensor, Tensor, Tensor, Tensor, Tensor, bool);

Tensor volume_render_cuvol(SparseGridSpec &, RaysSpec &, RenderOptions &);
Tensor volume_render_cuvol_image(SparseGridSpec &, CameraSpec &, RenderOptions &);

void volume_render_cuvol_backward(SparseGridSpec &, RaysSpec &, RenderOptions &, Tensor, Tensor, GridOutputGrads &);
void volume_render_cuvol_fused(SparseGridSpec &, RaysSpec &, RenderOptions &, Tensor, float, float, Tensor, GridOutputGrads &);

// Expected termination (depth) rendering
torch::Tensor volume_render_expected_term(SparseGridSpec &, RaysSpec &, RenderOptions &);
void volume_render_fused_rgbd(SparseGridSpec&, RaysSpec&, RenderOptions&, Tensor, Tensor, float, float, float, Tensor, Tensor, GridOutputGrads&);

// Depth rendering based on sigma-threshold as in Dex-NeRF
torch::Tensor volume_render_sigma_thresh(SparseGridSpec &, RaysSpec &, RenderOptions &, float);

// Pose estimation
torch::Tensor render_pos_forward(SparseGridSpec&, RaysSpec&, RenderOptions& , Tensor); 
torch::Tensor render_pos_forward_rgbd(SparseGridSpec&, RaysSpec&, RenderOptions& , Tensor, Tensor); 
void render_pos_backward(SparseGridSpec&, RaysSpec&, RenderOptions& , Tensor, Tensor, PoseGrads&); 
void render_pos_backward_rgbd(SparseGridSpec&, RaysSpec&, RenderOptions& , Tensor, Tensor, Tensor, Tensor, PoseGrads&); 

// Misc
Tensor dilate(Tensor);
void accel_dist_prop(Tensor);
void grid_weight_render(Tensor, CameraSpec &, float, float, bool, Tensor, Tensor, Tensor);

// Loss
Tensor tv(Tensor, Tensor, int, int, bool, float, bool);
void tv_grad(Tensor, Tensor, int, int, float, bool, float, bool, Tensor);
void tv_grad_sparse(Tensor, Tensor, Tensor, Tensor, int, int, float, bool, float, bool, bool, Tensor);
void msi_tv_grad_sparse(Tensor, Tensor, Tensor, Tensor, float, float, Tensor);

// Optim
void rmsprop_step(Tensor, Tensor, Tensor, Tensor, float, float, float, float, float);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#define _REG_FUNC(funname) m.def(#funname, &funname)
  _REG_FUNC(sample_grid);
  _REG_FUNC(sample_grid_backward);
  _REG_FUNC(volume_render_cuvol);
  _REG_FUNC(volume_render_cuvol_image);
  _REG_FUNC(volume_render_cuvol_backward);
  _REG_FUNC(volume_render_cuvol_fused);
  _REG_FUNC(volume_render_expected_term);
  _REG_FUNC(volume_render_fused_rgbd);
  _REG_FUNC(volume_render_sigma_thresh);

  _REG_FUNC(render_pos_forward);
  _REG_FUNC(render_pos_forward_rgbd);
  _REG_FUNC(render_pos_backward);
  _REG_FUNC(render_pos_backward_rgbd);

  // Loss
  _REG_FUNC(tv);
  _REG_FUNC(tv_grad);
  _REG_FUNC(tv_grad_sparse);
  _REG_FUNC(msi_tv_grad_sparse);

  // Misc
  _REG_FUNC(dilate);
  _REG_FUNC(accel_dist_prop);
  _REG_FUNC(grid_weight_render);

  // Optimizer
  _REG_FUNC(rmsprop_step);
#undef _REG_FUNC

  py::class_<SparseGridSpec>(m, "SparseGridSpec")
      .def(py::init<>())
      .def_readwrite("density_data", &SparseGridSpec::density_data)
      .def_readwrite("sh_data", &SparseGridSpec::sh_data)
      .def_readwrite("links", &SparseGridSpec::links)
      .def_readwrite("_offset", &SparseGridSpec::_offset)
      .def_readwrite("_scaling", &SparseGridSpec::_scaling)
      .def_readwrite("sh_dim", &SparseGridSpec::sh_dim);

  py::class_<CameraSpec>(m, "CameraSpec")
      .def(py::init<>())
      .def_readwrite("c2w", &CameraSpec::c2w)
      .def_readwrite("fx", &CameraSpec::fx)
      .def_readwrite("fy", &CameraSpec::fy)
      .def_readwrite("cx", &CameraSpec::cx)
      .def_readwrite("cy", &CameraSpec::cy)
      .def_readwrite("width", &CameraSpec::width)
      .def_readwrite("height", &CameraSpec::height);

  py::class_<RaysSpec>(m, "RaysSpec")
      .def(py::init<>())
      .def_readwrite("origins", &RaysSpec::origins)
      .def_readwrite("dirs", &RaysSpec::dirs);

  py::class_<RenderOptions>(m, "RenderOptions")
      .def(py::init<>())
      .def_readwrite("empty_space_brightness",
                     &RenderOptions::empty_space_brightness)
      .def_readwrite("step_size", &RenderOptions::step_size)
      .def_readwrite("sigma_thresh", &RenderOptions::sigma_thresh)
      .def_readwrite("stop_thresh", &RenderOptions::stop_thresh)
      .def_readwrite("near_clip", &RenderOptions::near_clip)
      .def_readwrite("use_spheric_clip", &RenderOptions::use_spheric_clip)
      .def_readwrite("last_sample_opaque", &RenderOptions::last_sample_opaque);

  py::class_<GridOutputGrads>(m, "GridOutputGrads")
      .def(py::init<>())
      .def_readwrite("grad_density_out", &GridOutputGrads::grad_density_out)
      .def_readwrite("grad_sh_out", &GridOutputGrads::grad_sh_out)
      .def_readwrite("mask_out", &GridOutputGrads::mask_out);

  py::class_<PoseGrads>(m, "PoseGrads")
      .def(py::init<>())
      .def_readwrite("grad_pose_origin_out", &PoseGrads::grad_pose_origin_out)
      .def_readwrite("grad_pose_direction_out", &PoseGrads::grad_pose_direction_out);
}


