/*

Copyright 2026 Nikolaos Tsapanos, see the LICENSE file for specifics.

This file contains the CUDA kernels that carry out the forward pass computation.

NOTES

1) For the math involved, please refer to:
  - https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf
  - https://geond.tech/wp-content/uploads/2026/07/PNDNN.pdf

2) In this implementation, the type T can be either float or double.

3) atomicAdd() is used to ensure correctness, as more than one thread could potentially attempt to write in the same memory position. Shared memory can be used to avoid it (as later implementation do), however it is a good starting point and not as bad as one might think.

*/

#include <cuda.h>
#include <cuda_runtime.h>
#ifdef DIVCEIL
#undef DIVCEIL
#endif
#define DIVCEIL(x,y) ((x+y-1)/y)

template <typename T>
__global__ void ParaConv2DInitCudaKernelNCHW(const T* __restrict__ h0, const T* __restrict__ p0, const int n, const int out_height, const int out_width, const int m, const int length, T* hsum, T* output) {

// Perform initialization.

  if (blockIdx.x<n) {
    if (blockIdx.y<m) {
      if (blockIdx.z<out_height) {
    
        int starti = threadIdx.x*length;
        int endi = (threadIdx.x+1)*length;
        if (endi>out_width) {
          endi=out_width;
        }//if endi
        for (int i = starti; i<endi; i++) {
          hsum[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + blockIdx.z*out_width + i] = __ldg(h0+blockIdx.y);
          output[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + blockIdx.z*out_width + i] = -(__ldg(p0+blockIdx.y)*__ldg(p0+blockIdx.y));
        }//for i
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x
  
}//ParaboloidInitCudaKernelNCHW
template __global__ void ParaConv2DInitCudaKernelNCHW<float>(const float* h0, const float* p0, const int n, const int out_height, const int out_width, const int m, const int length, float* hsum, float* output);
template __global__ void ParaConv2DInitCudaKernelNCHW<double>(const double* h0, const double* p0, const int n, const int out_height, const int out_width, const int m, const int length, double* hsum, double* output);



template <typename T>
__global__ void ParaConv2DHsumCudaKernelNCHW(const T* __restrict__ input, const T* __restrict__ h, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, T* hsum) {

// Compute Hsum (the signed, unregularized distance from the directrix hyperplane) and store it separately, because we will need it for the backward pass.

  if (blockIdx.x<n) {
    if (blockIdx.y<out_height) {
      if (blockIdx.z<out_width) {
        if (threadIdx.y<m) {
          T tmp = 0;
          if (threadIdx.x<channels) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                tmp += h[ threadIdx.y*channels*kernel_size0*kernel_size1 + threadIdx.x*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] * input[blockIdx.x*channels*height*width + threadIdx.x*height*width + ((blockIdx.y*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((blockIdx.z*strides1)+((y2+(kernel_size1/2))*dilation_rate1))];
              }//for y2
            }//for x2
          }//if threadIdx.x
          atomicAdd(hsum+(blockIdx.x*m*out_height*out_width + threadIdx.y*out_height*out_width + blockIdx.y*out_width + blockIdx.z), tmp);
        }//if threadIdx.y
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x



}//ParaboloidHsumCudaKernelNCHW
template __global__ void ParaConv2DHsumCudaKernelNCHW<float>(const float* input, const float* h, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, float* hsum);
template __global__ void ParaConv2DHsumCudaKernelNCHW<double>(const double* input, const double* h, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, double* hsum);



template <typename T>
__global__ void ParaConv2DPsqCudaKernelNCHW(const T* __restrict__ input, const T* __restrict__ p, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, T* output) {

// Compute Psq (the squared distance from the focal point) and store its negative in the output.

  if (blockIdx.x<n) {
    if (blockIdx.y<out_height) {
      if (blockIdx.z<out_width) {
        if (threadIdx.y<m) {
          T tmp1 = 0;
          for (int k=0; k<channels; k++) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                T tmp2 = input[blockIdx.x*channels*height*width + k*height*width + ((blockIdx.y*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((blockIdx.z*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] - p[ threadIdx.y*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2];
                tmp1-=tmp2*tmp2;
              }//for y2
            }//for x2
          }//for k
          atomicAdd(output+(blockIdx.x*m*out_height*out_width + threadIdx.y*out_height*out_width + blockIdx.y*out_width + blockIdx.z), tmp1);
        }//if threadIdx.y
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaboloidPsqCudaKernelNCHW
template __global__ void ParaConv2DPsqCudaKernelNCHW<float>(const float* input, const float* p, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, float* output);
template __global__ void ParaConv2DPsqCudaKernelNCHW<double>(const double* input, const double* p, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, double* output);



template <typename T>
__global__ void ParaConv2DOutputCudaKernelNCHW(const T factor, const int n, const int out_height, const int out_width, const int m, const int length, const T* __restrict__ hsum, T* output) {

// Per element add the squared Hsum to the output (that already contains the negative of Psq) to obtain the final output.

  if (blockIdx.x<n) {
    if (blockIdx.y<m) {
      if (blockIdx.z<out_height) {
        int starti = threadIdx.x*length;
        int endi = (threadIdx.x+1)*length;
        if (endi>out_width) {
          endi=out_width;
        }//if endi
        for (int i = starti; i<endi; i++) {
          T tmp = __ldg(hsum + (blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + blockIdx.z*out_width + i));
          output[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + blockIdx.z*out_width + i] += tmp * tmp;
          output[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + blockIdx.z*out_width + i] *= factor;
        }//for i
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaboloidOutputCudaKernelNCHW
template __global__ void ParaConv2DOutputCudaKernelNCHW<float>(const float factor, const int n, const int out_height, const int out_width, const int m, const int length, const float* hsum, float* output);
template __global__ void ParaConv2DOutputCudaKernelNCHW<double>(const double factor, const int n, const int out_height, const int out_width, const int m, const int length, const double* hsum, double* output);

