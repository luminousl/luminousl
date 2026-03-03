#include <stdio.h>
#include <stdlib.h>


__global__ void fast_bev_kernel(int n, int h, int w, int c, 
                                int n_ref, int bev_volume,
                                const float *__restrict__ imgs,
                                const int *__restrict__ img_coors,
                                const int *__restrict__ bev_coors,
                                float *bev_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int index = idx / c;
  int cur_c = idx % c;
  if (index >= n_ref) return;
  const int* img_coor = img_coors + index * 4;
  const int* bev_coor = bev_coors + index * 2;
  int img_idx = (img_coor[0] * n * h * w + img_coor[1] * h * w + img_coor[2] * w + img_coor[3]) * c + cur_c;
  int bev_idx = (bev_coor[0] * bev_volume + bev_coor[1]) * c + cur_c;
  const float pixel = imgs[img_idx];
  float* bev_ptr = bev_out + bev_idx;
  atomicAdd(bev_ptr, pixel);
}


void fast_bev(int n, int h, int w, int c, int n_ref, int bev_volume,
              const float* imgs, const int* img_coors,
              const int* bev_coors, float* bev_out) {
  int block_size = 256;
  int grid_size = static_cast<int>(ceil(static_cast<double>(n_ref) * c / block_size));
  fast_bev_kernel<<<grid_size, block_size>>>(
    n, h, w, c, n_ref, bev_volume, imgs, img_coors, bev_coors, bev_out);
}
