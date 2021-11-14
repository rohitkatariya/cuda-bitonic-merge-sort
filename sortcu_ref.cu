#include <iostream>
#include <limits>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sortcu.h"

__global__ void sortcu(uint32_t *data, int ndata);

__device__ int d_step;
__device__ int d_substep;

// check cuda error
void check_cuda_error(cudaError_t err, int line) {
  if (err != cudaError_t::cudaSuccess) {
    std::cerr << "CUDA error at line " << line << " : "
              << cudaGetErrorString(err) << "\n";
    std::exit(1);
  }
}

#define CHECK_ERROR(err) check_cuda_error(err, __LINE__)

// sets up and calls GPU kernel
void sort(uint32_t *data, int ndata) {
  uint32_t *h_data = data;
  const int num_data = ndata;
  const int num_bytes = num_data * sizeof(uint32_t);

  int least_pow2 = 0;
  while ((1ULL << least_pow2) < num_data) {
    least_pow2++;
  }

  const int padded_num_data = (1 << least_pow2);
  const int padded_num_bytes = padded_num_data * sizeof(uint32_t);
  const int pad_num_data = padded_num_data - num_data;

  uint32_t *d_data = nullptr;
  CHECK_ERROR(cudaMalloc(&d_data, padded_num_bytes));
  CHECK_ERROR(cudaMemcpy(d_data + pad_num_data, h_data, num_bytes,
                    cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemset(d_data, 0, pad_num_data));

  int num_threads = 512;
  int num_blocks = (padded_num_data + num_threads - 1) / num_threads;

  for (int step = 2; step <= padded_num_data; step <<= 1) {
    for (int substep = step >> 1; substep > 0; substep >>= 1) {
      CHECK_ERROR(cudaMemcpyToSymbol(d_step, &step, sizeof(int)));
      CHECK_ERROR(cudaMemcpyToSymbol(d_substep, &substep, sizeof(int)));
      sortcu<<<num_blocks, num_threads>>>(d_data, padded_num_data);
    }
  }

  CHECK_ERROR(cudaDeviceSynchronize());

  CHECK_ERROR(cudaMemcpy(h_data, d_data + pad_num_data, num_bytes,
                    cudaMemcpyDeviceToHost));

  CHECK_ERROR(cudaFree(d_data));
}

// kernel ran on the GPU
__global__ void sortcu(uint32_t *data, int ndata) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = i ^ d_substep;

  uint32_t data_i = data[i];
  uint32_t data_j = data[j];

  if (j > i) {
    if ((((d_step & i) == 0) && (data_i > data_j)) ||
        (((d_step & i) != 0) && (data_i < data_j))) {
      data[i] = data_j;
      data[j] = data_i;
    }
  }
}
