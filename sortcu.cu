#include <iostream>
#include <limits>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sortcu.h"

// #define DEBUG

#define MOD_MAX_MY 4294967295
// #define MOD_MAX_MY 10

#define MOD_MAX_SUM_MY(a,b) ( (long(a)+long(b)) % MOD_MAX_MY )

__global__ void sortcu( uint32_t *data_arr,uint32_t *prefix_arr, int ndata);
__global__ void prefix_up(uint32_t *B_this, uint32_t *B_prev, int num_ele_this);
__global__ void prefix_down(uint32_t *B_h, uint32_t *C_h, uint32_t *C_hp1, int num_ele_this) ;
__global__ void init_idx_arr(uint32_t *index_arr,int num_ele_this);
__global__ void d_idx_convert(uint32_t *index_arr,uint32_t * d_prfx ,int num_ele_this);
__device__ int64_t d_step;
__device__ int64_t d_substep;
using namespace std;
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
  uint32_t *h_data = data; //host pointer
  const int num_data = ndata;
  const long num_bytes = num_data * sizeof(uint32_t);

  int least_pow2 = 0;
  while ((1ULL << least_pow2) < num_data) {
    least_pow2++;
  }
  // printf("\nnum_data:%d,lp:%d\n",num_data,least_pow2);
  
  const int padded_num_data = (1 << least_pow2); //number of elements after padding
  const long padded_num_bytes = padded_num_data * sizeof(uint32_t); // number of bytes with padding
  const int pad_num_data = padded_num_data - num_data; // number of elements to be padded
 
  uint32_t **B = new uint32_t*[least_pow2+1];
  uint32_t **C = new uint32_t*[least_pow2+1];
  uint32_t *this_addr=nullptr;
  
  CHECK_ERROR(cudaMalloc(&this_addr, padded_num_bytes));
  B[0]=this_addr;
  CHECK_ERROR(cudaMalloc(&this_addr, padded_num_bytes));
  C[0]=this_addr;

  // cout<<"\nnum_bytes:"<<num_bytes;
  CHECK_ERROR(cudaMemcpy(B[0] + pad_num_data, h_data, num_bytes,
                    cudaMemcpyHostToDevice)); //copy all elements to device after leaving pad number of elements
  CHECK_ERROR(cudaMemset(B[0], 0, pad_num_data)); 
  // CHECK_ERROR(cudaMemcpy(C[0] + pad_num_data, h_data, num_bytes,
  //                   cudaMemcpyHostToDevice)); //copy all elements to device after leaving pad number of elements
  // CHECK_ERROR(cudaMemset(C[0], 0, pad_num_data)); 
  
  int num_ele_this = padded_num_data;
  
  int num_threads = 512;
  int num_blocks = (padded_num_data + num_threads - 1) / num_threads;
  
  // uint32_t *temp_B = new uint32_t[padded_num_data];
  // cout<<"\nData Flow up";
  // Data Flow up
  for(int h = 1 ; h<=least_pow2;h++){
    num_ele_this = num_ele_this/2;
    // printf("\nallocating %d",num_ele_this);
    CHECK_ERROR(cudaMalloc(&this_addr, long(num_ele_this) * long(sizeof(uint32_t))));
    B[h]=this_addr;
    CHECK_ERROR(cudaMalloc(&this_addr, long(num_ele_this) * long(sizeof(uint32_t))));
    C[h]=this_addr;
    // CHECK_ERROR(cudaMemcpyToSymbol(d_num_ele_this, &num_ele_this, sizeof(int)));
    prefix_up<<<num_blocks, num_threads>>>(B[h], B[h-1],num_ele_this);
    // cout<<"\nh:"<<h;
  }  
  
  CHECK_ERROR(cudaDeviceSynchronize());
  // Data Flow Down
  // cout<<"\nData Flow Down";

  num_ele_this=1;
  prefix_down<<<num_blocks, num_threads>>>(B[least_pow2],C[least_pow2],nullptr,num_ele_this);
  for(int h = least_pow2-1;h>=0;h--){
    num_ele_this*=2;
    prefix_down<<<num_blocks, num_threads>>>(B[h],C[h],C[h+1],num_ele_this);
  }
  CHECK_ERROR(cudaDeviceSynchronize());
  // #ifdef DEBUG
  //   num_ele_this = 1;
  // uint32_t *temp_B = new uint32_t[padded_num_data];

  //   for(int h = least_pow2 ; h>=0;h--){
  //     CHECK_ERROR(cudaMemcpy(temp_B, C[h], num_ele_this*sizeof(uint32_t),
  //                     cudaMemcpyDeviceToHost));
  //     cout<<"\n\n["<<h<<"]";
  //     for(int i =0;i<num_ele_this;i++)
  //       cout<<" "<<temp_B[i];
  //     num_ele_this = num_ele_this*2;
  //   }
  // #endif

  // free temp arrays
  for(int h = 1 ; h<=least_pow2;h++){
    CHECK_ERROR(cudaFree(B[h]));
    CHECK_ERROR(cudaFree(C[h]));
  }
  // cout<<"\nprefix done";
  uint32_t *d_prefix_arr = C[0];
  uint32_t *d_data_arr = B[0];
  // uint32_t *prefix_arr= new uint32_t[num_data];
  // CHECK_ERROR(cudaMemcpy(prefix_arr, C[0]+pad_num_data, num_data*sizeof(uint32_t),
  //                     cudaMemcpyDeviceToHost));
  
  // uint32_t *index_arr=nullptr;
  // CHECK_ERROR(cudaMalloc(&index_arr, padded_num_bytes));
  
  // init_idx_arr<<<num_blocks, num_threads>>>(index_arr,num_ele_this);
  CHECK_ERROR(cudaDeviceSynchronize());
  // Sorting 

  for (int64_t step = 2; step <= padded_num_data; step <<= 1) {
    for (int64_t substep = step >> 1; substep > 0; substep >>= 1) {
      // printf("\nsteps:%d,%d",step,substep);
      CHECK_ERROR(cudaMemcpyToSymbol(d_step, &step, sizeof(int64_t)));
      CHECK_ERROR(cudaMemcpyToSymbol(d_substep, &substep, sizeof(int64_t)));
      sortcu<<<num_blocks, num_threads>>>(d_data_arr,d_prefix_arr, padded_num_data);
  // CHECK_ERROR(cudaDeviceSynchronize());

    }
  }
  
  
  // printf("\nsorting done");
  
  CHECK_ERROR(cudaDeviceSynchronize());
  CHECK_ERROR(cudaMemcpy(h_data, d_data_arr+pad_num_data, num_data*sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));

  
  // #ifdef DEBUG
  // uint32_t *temp_B = new uint32_t[padded_num_data];
  // uint32_t *prefix_arr= new uint32_t[num_data];
  // cout<<"\nprefix_orig";
  // for(int i =0;i<num_data;i++)
  //   cout<<" "<<prefix_arr[i];
    // cout<<"\n\norig:";
    // for(int i =0;i<num_data;i++)
    //   cout<<" "<<h_data[i]<<"_"<<prefix_arr[i];

    
    
    
    // CHECK_ERROR(cudaMemcpy(prefix_arr, d_prefix_arr+pad_num_data, num_data*sizeof(uint32_t),
    //                     cudaMemcpyDeviceToHost));
    // cout<<"\n\nsorted:";
    // for(int i =0;i<min(num_data,100);i++)
    // for(int i =0;i<num_data;i++){
    //   if(i>0 && prefix_arr[i]==prefix_arr[i-1]){
    //         continue;
    //     }
    //     if(i<num_data-1  && prefix_arr[i]==prefix_arr[i+1]){
    //         continue;
    //     }
    //   cout<<h_data[i]<<"_"<<prefix_arr[i]<<" ";
    //   if(i%10==0){
    //     cout<<"\n";
    //   }
    // }

    // CHECK_ERROR(cudaMemcpy(temp_B, index_arr, long(padded_num_data)*sizeof(uint32_t),
    //                     cudaMemcpyDeviceToHost));
    // cout<<"\n\nindex   :";
    // for(int i =0;i<padded_num_data;i++)
    //   cout<<" "<<temp_B[i];
  // #endif
  CHECK_ERROR(cudaFree(B[0]));
  CHECK_ERROR(cudaFree(C[0]));

  delete[] C;
  delete[] B;
  
}

// kernel ran on the GPU
__global__ void d_idx_convert(uint32_t *index_arr,uint32_t * d_arr ,int num_ele_this) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<num_ele_this){
    index_arr[i]=d_arr  [index_arr[i]];
  }
  
}

// kernel ran on the GPU
__global__ void init_idx_arr(uint32_t *index_arr,int num_ele_this) {
  uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<num_ele_this){
    index_arr[i]=i;
  }
  
}

// kernel ran on the GPU
__global__ void prefix_down(uint32_t *B_h, uint32_t *C_h, uint32_t *C_hp1, int num_ele_this) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i==0)
    C_h[0]=B_h[0];
  else if(i<num_ele_this && i%2==1){
    C_h[i]=C_hp1[i/2];
  }
  else if(i<num_ele_this && i%2==0){
    C_h[i]=MOD_MAX_SUM_MY(C_hp1[i/2-1],B_h[i]);
  }
}

// kernel ran on the GPU
__global__ void prefix_up(uint32_t *B_this, uint32_t *B_prev, int num_ele_this) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<num_ele_this){
    B_this[i]= MOD_MAX_SUM_MY(B_prev[2*i],B_prev[2*i+1] ) ;
  }
} 

// kernel ran on the GPU
__global__ void sortcu(uint32_t *d_data_arr, uint32_t *d_prefix_arr, int ndata) {
  int64_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i<ndata){
    int64_t j = i ^ d_substep;

    uint32_t data_i = d_prefix_arr[i];
    uint32_t data_j = d_prefix_arr[j];
    uint32_t vdata_i = d_data_arr[i];
    uint32_t vdata_j = d_data_arr[j];
    // uint32_t temp_my = 0;
    if (j > i) {
      if ((((d_step & i) == 0) && (data_i > data_j)) ||
          (((d_step & i) != 0) && (data_i < data_j))) {
        d_prefix_arr[i] = data_j;
        d_prefix_arr[j] = data_i;
        d_data_arr[i]=vdata_j;
        d_data_arr[j]=vdata_i;
      }
    }
  }
}

// // kernel ran on the GPU
// __global__ void sortcu(uint32_t *d_data_arr, uint32_t *d_prefix_arr, int ndata) {
//   int i = blockDim.x * blockIdx.x + threadIdx.x;
//   int j = i ^ d_substep;

//   uint32_t data_i = d_prefix_arr[i];
//   uint32_t data_j = d_prefix_arr[j];
//   uint32_t temp_my = 0;
//   if (j > i) {
//     if ((((d_step & i) == 0) && (data_i > data_j)) ||
//         (((d_step & i) != 0) && (data_i < data_j))) {
//       d_prefix_arr[i] = data_j;
//       d_prefix_arr[j] = data_i;
//       temp_my = d_data_arr[i];
//       d_data_arr[i] = d_data_arr[j];
//       d_data_arr[j]=temp_my;

//     }
//   }
// }


// // kernel ran on the GPU
// __global__ void sortcu1(uint32_t *idx_arr, uint32_t *prefix_arr, int ndata) {
//   int i = blockDim.x * blockIdx.x + threadIdx.x;
//   int j = i ^ d_substep;
 
//   uint32_t data_i = prefix_arr[idx_arr[i]];
//   uint32_t data_j = prefix_arr[idx_arr[j]];
//   uint32_t data_i_i = idx_arr[i];
//   uint32_t data_j_i = idx_arr[j];

//   if (j > i) {
//     if ((((d_step & i) == 0) && (data_i > data_j)) ||
//         (((d_step & i) != 0) && (data_i < data_j))) {
//       idx_arr[i] = data_i_i;
//       idx_arr[j] = data_j_i;
//     }
//   }
// }
