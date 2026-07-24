/*

Copyright 2026 Nikolaos Tsapanos, see the LICENSE file for specifics.

This file contains the CUDA kernels that carry out the backward pass computation.

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
__global__ void ParaConv2DGradZeroTensorCudaKernel(int n1, int n2, int length, T* tftensor) {
  
  if (blockIdx.x<n1) {
    int starti = threadIdx.x*length;
    int endi = (threadIdx.x+1)*length;
    if (endi>n2) {
      endi=n2;
    }//if endi
    for (int i = starti; i<endi; i++) {
      tftensor[blockIdx.x*n2 + i] = 0;
    }//for i
  }//if blockIdx.x
  
}//ParaConv2DGradZeroTensorCudaKernel
template __global__ void ParaConv2DGradZeroTensorCudaKernel<float>(int n1, int n2, int length, float* tftensor);
template __global__ void ParaConv2DGradZeroTensorCudaKernel<double>(int n1, int n2, int length, double* tftensor);



template <typename T>
__global__ void ParaConv2DGradInputNCHWKernel(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ h, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* inputgrad) {

// This kernel computes the backwards gradient.

  if (blockIdx.x<n) {
    if (blockIdx.y<out_height) {
      if (blockIdx.z<out_width) {
        if (threadIdx.x<channels) {
          for (int ix2=0; ix2<kernel_size0; ix2++) {
            int x2 = (kernel_size0/2) - ix2;
            for (int iy2=0; iy2<kernel_size1; iy2++) {
              int y2 = (kernel_size1/2) - iy2;
              T inputgradtmp = 0;
              for (int j=0;j<m;j++) {
                inputgradtmp += grad[blockIdx.x*m*out_height*out_width + j*out_height*out_width + blockIdx.y*out_width + blockIdx.z] * ( ( hsum[blockIdx.x*m*out_height*out_width + j*out_height*out_width + blockIdx.y*out_width + blockIdx.z] *  h[j*channels*kernel_size0*kernel_size1 + threadIdx.x*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2]) -  input[ blockIdx.x*channels*height*width + threadIdx.x*height*width + ((blockIdx.y*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((blockIdx.z*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] + p[j*channels*kernel_size0*kernel_size1 + threadIdx.x*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] );
              }//for j
              atomicAdd(inputgrad+(blockIdx.x*channels*height*width + threadIdx.x*height*width + ((blockIdx.y*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((blockIdx.z*strides1)+((y2+(kernel_size1/2))*dilation_rate1))), inputgradtmp);
            }//for iy2
          }//for ix2
        }//if 
      }//if blockIdx.z
    }//if blockIdx.y
  }//if if blockIdx.x

}//ParaConv2DGradInputNCHWKernel
template __global__ void ParaConv2DGradInputNCHWKernel<float>(const float* grad, const float* hsum, const float* h, const float* p, const float* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, float* inputgrad);
template __global__ void ParaConv2DGradInputNCHWKernel<double>(const double* grad, const double* hsum, const double* h, const double* p, const double* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, double* inputgrad);



template <typename T>
__global__ void ParaConv2DGradInputKernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ h, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* inputgrad) {

// This is an older kernel that computes the backwards gradient. There was an issue in which it run into the 1024 threads per block CUDA limit. It is currently unused and only included for educational purposes.

  if (blockIdx.x<n) {
    if (blockIdx.y<channels) {
      if (blockIdx.z<out_height) {
        if (threadIdx.x<out_width) {
          if (threadIdx.y<m) {
            if (threadIdx.z<kernel_size0) {
              for (int tmpx2=0; tmpx2<kernel_size0; tmpx2++) {
                int x2 = tmpx2 - kernel_size1/2;
                for (int tmpy2=0; tmpy2<kernel_size1; tmpy2++) {
                  int y2 = tmpy2 - kernel_size1/2;
                
                  atomicAdd(inputgrad+ (blockIdx.x*channels*height*width + blockIdx.y*height*width + ((blockIdx.z*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((threadIdx.x*strides1)+((y2+(kernel_size1/2))*dilation_rate1))), 
                 __ldg(grad + (blockIdx.x*m*out_height*out_width + threadIdx.y*out_height*out_width + blockIdx.z*out_width + threadIdx.x)) *
                 ( ( __ldg( hsum + (blockIdx.x*m*out_height*out_width + threadIdx.y*out_height*out_width + blockIdx.z*out_width + threadIdx.x)) *
                  __ldg(h + (threadIdx.y*channels*kernel_size0*kernel_size1 + blockIdx.y*kernel_size0*kernel_size1 + (x2+(kernel_size0/2))*kernel_size1 + (y2+(kernel_size1/2)))) ) -
                   __ldg(input +( blockIdx.x*channels*height*width + blockIdx.y*height*width + ((blockIdx.z*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((threadIdx.x*strides1)+((y2+(kernel_size1/2))*dilation_rate1)))) +
                    __ldg(p + (threadIdx.y*channels*kernel_size0*kernel_size1 + blockIdx.y*kernel_size0*kernel_size1 + (x2+(kernel_size0/2))*kernel_size1 + (y2+(kernel_size1/2)))) )
                      );          
                }//for y2
              }//for x2
            }//if threadIdx.z
          }//if threadIdx.y
        }//if threadIdx.x
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaConv2DGradInputKernelNCHW
template __global__ void ParaConv2DGradInputKernelNCHW<float>(const float* grad, const float* hsum, const float* h, const float* p, const float* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, float* inputgrad);
template __global__ void ParaConv2DGradInputKernelNCHW<double>(const double* grad, const double* hsum, const double* h, const double* p, const double* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, double* inputgrad);



template <typename T>
__global__ void ParaConv2DGradHKernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* hgrad) {

// Compute the gradient for the directrix hyperplane parameters.

  if (blockIdx.x<n) {
    if (blockIdx.y<m) {
      if (blockIdx.z<channels) {
        if (threadIdx.x<out_height) {
          for (int y1=0;y1<out_width;y1++) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                atomicAdd( hgrad + (blockIdx.y*channels*kernel_size0*kernel_size1 + blockIdx.z*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2), grad[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + threadIdx.x*out_width + y1] * hsum[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + threadIdx.x*out_width + y1] * input[ blockIdx.x*channels*height*width + blockIdx.z*height*width + ((threadIdx.x*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))]);
              }//for iy2
            }//for ix2
          }//for y1
        }//if threadIdx.x
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaConv2DGradHKernelNCHW
template __global__ void ParaConv2DGradHKernelNCHW<float>(const float* grad, const float* hsum, const float* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, float* hgrad);
template __global__ void ParaConv2DGradHKernelNCHW<double>(const double* grad, const double* hsum, const double* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, double* hgrad);



template <typename T>
__global__ void ParaConv2DGradH0KernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const int n, const int m, int out_height, int out_width, const int length, T* h0grad) {

// Compute the gradient for the directrix hyperplane bias.

  if (blockIdx.x<n) {
    if (blockIdx.y<out_height) {
      if (blockIdx.z<m) {
        if (threadIdx.x<out_width) {
          atomicAdd(h0grad + blockIdx.z, grad[blockIdx.x*m*out_height*out_width + blockIdx.z*out_height*out_width + blockIdx.y*out_width + threadIdx.x] * hsum[blockIdx.x*m*out_height*out_width + blockIdx.z*out_height*out_width + blockIdx.y*out_width + threadIdx.x]);
        }//if threadIdx.x
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaConv2DGradH0KernelNCHW
template __global__ void ParaConv2DGradH0KernelNCHW<float>(const float* grad, const float* hsum, const int n, const int m, int out_height, int out_width, const int length, float* h0grad);
template __global__ void ParaConv2DGradH0KernelNCHW<double>(const double* grad, const double* hsum, const int n, const int m, int out_height, int out_width, const int length, double* h0grad);



template <typename T>
__global__ void ParaConv2DGradPKernelNCHW(const T* __restrict__ grad, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* pgrad) {

// Compute the gradient for the focal point parameters.

  if (blockIdx.x<n) {
    if (blockIdx.y<m) {
      if (blockIdx.z<channels) {
        if (threadIdx.x<out_height) {
          for (int y1=0;y1<out_width;y1++) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                atomicAdd(pgrad + (blockIdx.y*channels*kernel_size0*kernel_size1 + blockIdx.z*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2), grad[blockIdx.x*m*out_height*out_width + blockIdx.y*out_height*out_width + threadIdx.x*out_width + y1] * ( input[ blockIdx.x*channels*height*width + blockIdx.z*height*width + ((threadIdx.x*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] - p[blockIdx.y*channels*kernel_size0*kernel_size1 + blockIdx.z*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2]));
              }//for iy2
            }//for ix2
          }//for y1
        }//if threadIdx.x
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaConv2DGradPKernelNCHW
template __global__ void ParaConv2DGradPKernelNCHW<float>(const float* grad, const float* p, const float* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, float* pgrad);
template __global__ void ParaConv2DGradPKernelNCHW<double>(const double* grad, const double* p, const double* input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, double* pgrad);



template <typename T>
__global__ void ParaConv2DGradP0KernelNCHW(const T* __restrict__ grad, const T* __restrict__ p0, const int n, const int m, int out_height, int out_width, const int length, T* p0grad) {

// Compute the gradient for the extra dimension of the focal point.

  if (blockIdx.x<n) {
    if (blockIdx.y<out_height) {
      if (blockIdx.z<m) {
        if (threadIdx.x<out_width) {
          atomicAdd(p0grad + blockIdx.z, grad[blockIdx.x*m*out_height*out_width + blockIdx.z*out_height*out_width + blockIdx.y*out_width + threadIdx.x] * (-p0[blockIdx.z]));
        }//if threadIdx.x
      }//if blockIdx.z
    }//if blockIdx.y
  }//if blockIdx.x

}//ParaConv2DGradP0KernelNCHW
template __global__ void ParaConv2DGradP0KernelNCHW<float>(const float* grad, const float* p0, const int n, const int m, int out_height, int out_width, const int length, float* p0grad);
template __global__ void ParaConv2DGradP0KernelNCHW<double>(const double* grad, const double* p0, const int n, const int m, int out_height, int out_width, const int length, double* p0grad);



template <typename T>
__global__ void ParaConv2DGrad2TensorCudaKernel(int n1, int n2, int length, T* tensor) {
  
  if (blockIdx.x<n1) {
    int starti = threadIdx.x * length;
    int endi = (threadIdx.x + 1) * length;
    if (endi > n2) {
      endi = n2;
    }//if endi
    for (int i = starti; i<endi; i++) {
      tensor[blockIdx.x * n2 + i] = tensor[blockIdx.x * n2 + i]+tensor[blockIdx.x * n2 + i];
    }//for i
  }//if blockIdx.x
  
}//ParaConv2DGrad2TensorCudaKernel
template __global__ void ParaConv2DGrad2TensorCudaKernel<float>(int n1, int n2, int length, float* tensor);
template __global__ void ParaConv2DGrad2TensorCudaKernel<double>(int n1, int n2, int length, double* tensor);



template <typename T>
__global__ void ParaConv2DGrad2FactorTensorCudaKernel(const T factor, int n1, int n2, int length, T* tensor) {
  
  if (blockIdx.x < n1) {
    int starti = threadIdx.x * length;
    int endi = (threadIdx.x + 1) * length;
    if (endi > n2) {
      endi = n2;
    }//if endi
    for (int i = starti; i<endi; i++) {
      tensor[blockIdx.x * n2 + i] = factor*(tensor[blockIdx.x * n2 + i]+tensor[blockIdx.x * n2 + i]);
    }//for i
  }//if blockIdx.x
  
}//ParaConv2DGrad2FactorTensorCudaKernel
template __global__ void ParaConv2DGrad2FactorTensorCudaKernel<float>(const float factor, int n1, int n2, int length, float* tensor);
template __global__ void ParaConv2DGrad2FactorTensorCudaKernel<double>(const double factor, int n1, int n2, int length, double* tensor);

