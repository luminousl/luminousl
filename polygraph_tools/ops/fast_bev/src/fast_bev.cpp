// Copyright (c) Phigent Robotics. All rights reserved.
// Reference https://arxiv.org/abs/2211.17111
#include <torch/torch.h>
#include <c10/cuda/CUDAGuard.h>

// CUDA function declarations
void fast_bev(int n, int h, int w, int c, int n_ref, int bev_volume,
              const float* imgs, const int* img_coors,
              const int* bev_coors, float* bev_out);

void fast_bev_forward(
  int bev_volume,
  const at::Tensor imgs,
  const at::Tensor img_coors,
  const at::Tensor bev_coors,
  at::Tensor bev_out
) {
  int n = imgs.size(1);
  int h = imgs.size(2);
  int w = imgs.size(3);
  int c = imgs.size(4);
  int n_ref = img_coors.size(0);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(imgs));
  const float* imgs_ = imgs.data_ptr<float>();
  const int* img_coors_ = img_coors.data_ptr<int>();
  const int* bev_coors_ = bev_coors.data_ptr<int>();
  float* bev_out_ = bev_out.data_ptr<float>();

  fast_bev(n ,h, w, c, n_ref, bev_volume, imgs_, img_coors_, bev_coors_, bev_out_);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_bev_forward", &fast_bev_forward,
        "fast_bev_forward function");
}
