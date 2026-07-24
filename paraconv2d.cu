/*

Copyright 2026 Nikolaos Tsapanos, see the LICENSE file for specifics.

This file contains the code that calls the CUDA kernels that carry out the computation.

NOTES

1) For the math involved, please refer to:
  - https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf
  - https://geond.tech/wp-content/uploads/2026/07/PNDNN.pdf

2) In this implementation, the type T can be either float or double.

3) While this custom op has 4 parameters, h0, h, p and p0, they are all packed in a single vector in the listed order.

  To retrieve each parameter individually, we can do as follows:

    const scalar_t* h0 = parameters; // h0 is at the start of the tensor and has m elements.
    const scalar_t* h = h0+(m); // h starts m positions after h0 and has m*d elements.
    const scalar_t* p = h+(m*d); // p starts m*d positions after h and has m*d elements. 
    const scalar_t* p0 = p+(m*d); // p0 starts m*d positions after p and has m elements.

  The gradient tensor is packed in the same order.

    scalar_t* h0grad=parametersgrad;
    scalar_t* hgrad=h0grad+(m);
    scalar_t* pgrad=hgrad+(m*d);
    scalar_t* p0grad=pgrad+(m*d);

*/

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <chrono>

#ifdef DIVCEIL
#undef DIVCEIL
#endif
#define DIVCEIL(x,y) ((x+y-1)/y)



template <typename T>
__global__ void ParaConv2DInitCudaKernelNCHW(const T* __restrict__ h0, const T* __restrict__ p0, const int n, const int out_height, const int out_width, const int m, const int length, T* hsum, T* output);
template <typename T>
__global__ void ParaConv2DHsumCudaKernelNCHW(const T* __restrict__ input, const T* __restrict__ h, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, T* hsum);
template <typename T>
__global__ void ParaConv2DPsqCudaKernelNCHW(const T* __restrict__ input, const T* __restrict__ p, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, const int out_height, const int out_width, const int length, T* output);
template <typename T>
__global__ void ParaConv2DOutputCudaKernelNCHW(const T factor, const int n, const int out_height, const int out_width, const int m, const int length, const T* __restrict__ hsum, T* output);

template <typename T>
__global__ void ParaConv2DGradZeroTensorCudaKernel(int n1, int n2, int length, T* tftensor);
template <typename T>
__global__ void ParaConv2DGradHKernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* hgrad);
template <typename T>
__global__ void ParaConv2DGradInputKernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ h, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* inputgrad);
template <typename T>
__global__ void ParaConv2DGradH0KernelNCHW(const T* __restrict__ grad, const T* __restrict__ hsum, const int n, const int m, int out_height, int out_width, const int length, T* h0grad);
template <typename T>
__global__ void ParaConv2DGradPKernelNCHW(const T* __restrict__ grad, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* pgrad);
template <typename T>
__global__ void ParaConv2DGradP0KernelNCHW(const T* __restrict__ grad, const T* __restrict__ p0, const int n, const int m, int out_height, int out_width, const int length, T* p0grad);
template <typename T>
__global__ void ParaConv2DGrad2TensorCudaKernel(int n1, int n2, int length, T* tftensor);
template <typename T>
__global__ void ParaConv2DGrad2FactorTensorCudaKernel(const T factor, int n1, int n2, int length, T* tftensor);

template <typename T>
extern __global__ void ParaConv2DGradInputNCHWKernel(const T* __restrict__ grad, const T* __restrict__ hsum, const T* __restrict__ h, const T* __restrict__ p, const T* __restrict__ input, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, const int n, const int height, const int width, const int channels, const int m, int out_height, int out_width, const int length, T* inputgrad);



template <typename scalar_t>
void paraconv2d_forward_cuda_main(at::Device dev, const scalar_t* input, const scalar_t* parameters, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* output, scalar_t* hsum) {

// This if the forward pass implementation for CUDA.

  int d = channels*kernel_size0*kernel_size1;
  const scalar_t* h0 = parameters;
  const scalar_t* h = h0+(m);
  const scalar_t* p = h+(m*d);
  const scalar_t* p0 = p+(m*d);

  const scalar_t factor = (scalar_t)doublefactor;

  cudaStream_t paraconv2dstream;
  
  cudaStreamCreate(&paraconv2dstream);

// Perform initialization.
  dim3 initgrid(DIVCEIL(n,32)*32,DIVCEIL(m,32)*32,DIVCEIL(out_height,32)*32);
  int initlength = 8;
  int initnthreads = DIVCEIL(out_width, initlength);
  ParaConv2DInitCudaKernelNCHW<scalar_t><<<initgrid, initnthreads, 0, paraconv2dstream>>>(h0, p0, n, out_height, out_width, m, initlength, hsum, output);

// Compute Hsum (the signed, unregularized distance from the directrix hyperplane) and store it separately, because we will need it for the backward pass.
  dim3 hsumgrid(DIVCEIL(n,32)*32,DIVCEIL(out_height,32)*32,DIVCEIL(out_width,32)*32);
  int hsumlength = 8;  
  dim3 hsumthreads(DIVCEIL(channels,4)*4,DIVCEIL(m,4)*4,1);
  ParaConv2DHsumCudaKernelNCHW<scalar_t><<<hsumgrid, hsumthreads, 0, paraconv2dstream>>>(input, h, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, height, width, channels, m, out_height, out_width, hsumlength, hsum);

// Compute Psq (the squared distance from the focal point) and store its negative in the output.
  dim3 psqgrid(DIVCEIL(n,32)*32,DIVCEIL(out_height,32)*32,DIVCEIL(out_width,32)*32);
  int psqlength = 8;  
  dim3 psqthreads(1,DIVCEIL(m,4)*4,1);
  ParaConv2DPsqCudaKernelNCHW<scalar_t><<<psqgrid, psqthreads, 0, paraconv2dstream>>>(input, p, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, height, width, channels, m, out_height, out_width, psqlength, output);

// Per element add the squared Hsum to the output (that already contains the negative of Psq) to obtain the final output.
  dim3 outputgrid(DIVCEIL(n,32)*32,DIVCEIL(m,32)*32,DIVCEIL(out_height,32)*32);
  int outputlength = 8;
  int outputnthreads = DIVCEIL(out_width, outputlength);
  ParaConv2DOutputCudaKernelNCHW<scalar_t><<<outputgrid, outputnthreads, 0, paraconv2dstream>>>(factor, n, out_height, out_width, m, outputlength, hsum, output);

  cudaStreamDestroy(paraconv2dstream);

}//paraconv2d_forward_cuda_main

template void paraconv2d_forward_cuda_main<float>(at::Device dev, const float* input, const float* parameters, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, float* output, float* hsum);
template void paraconv2d_forward_cuda_main<double>(at::Device dev, const double* input, const double* parameters, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, double* output, double* hsum);



template <typename scalar_t>
void paraconv2d_backward_cuda_main(at::Device dev, const scalar_t* grad, const scalar_t* input, const scalar_t* parameters, const scalar_t* hsum, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* inputgrad, scalar_t* parametersgrad, bool skip_input_grad) {

// This if the backward pass implementation for CUDA.

  int d = channels*kernel_size0*kernel_size1;
  const scalar_t* h0 = parameters;
  const scalar_t* h = h0+(m);
  const scalar_t* p = h+(m*d);
  const scalar_t* p0 = p+(m*d);

  scalar_t* h0grad=parametersgrad;
  scalar_t* hgrad=h0grad+(m);
  scalar_t* pgrad=hgrad+(m*d);
  scalar_t* p0grad=pgrad+(m*d);
    
  const scalar_t factor = (scalar_t)doublefactor;


  cudaStream_t paraconv2dgradstream;
  
  cudaStreamCreate(&paraconv2dgradstream);
  int blocksize;

    if (!skip_input_grad) {

// Compute the backwards gradient. If this is the first layer after the input, this computation can be skipped.
      blocksize = n*height;
      int inputlength = DIVCEIL(width*channels,1024);
      int inputnthreads = DIVCEIL(width*channels,inputlength);
      ParaConv2DGradZeroTensorCudaKernel<scalar_t><<<blocksize, inputnthreads, 0, paraconv2dgradstream>>>(n*height, width*channels, inputlength, inputgrad);
      dim3 inputgradgrid(n,out_height,out_width);
      int inputgradlength = DIVCEIL(kernel_size0*kernel_size1*channels,1024);
      dim3 inputgradnthreads(channels,1,1);
      ParaConv2DGradInputNCHWKernel<scalar_t><<<inputgradgrid, inputgradnthreads, 0, paraconv2dgradstream>>>(grad, hsum, h, p, input, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, height, width, channels, m, out_height, out_width, inputgradlength, inputgrad);
    
    }//if !skip_input_grad

// Compute the gradient for the directrix hyperplane parameters.
  dim3 hgradgrid(DIVCEIL(n,32)*32,DIVCEIL(m,32)*32,DIVCEIL(channels,32)*32);
  dim3 hgradnthreads(out_height,1,1);
  ParaConv2DGradHKernelNCHW<scalar_t><<<hgradgrid, hgradnthreads, 0, paraconv2dgradstream>>>(grad, hsum, input, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, height, width, channels, m, out_height, out_width, 1, hgrad);

// Compute the gradient for the directrix hyperplane bias.
  dim3 h0gradgrid(DIVCEIL(n,32)*32,DIVCEIL(out_height,32)*32,DIVCEIL(m,32)*32);
  int h0length = 8;
  dim3 h0gradnthreads(out_width,1,1);
  ParaConv2DGradH0KernelNCHW<scalar_t><<<h0gradgrid, h0gradnthreads, 0, paraconv2dgradstream>>>(grad, hsum, n, m, out_height, out_width, 1, h0grad);

// Compute the gradient for the focal point parameters.
  dim3 pgradgrid(DIVCEIL(n,32)*32,DIVCEIL(m,32)*32,DIVCEIL(channels,32)*32);
  dim3 pgradnthreads(out_height,1,1);
  ParaConv2DGradPKernelNCHW<scalar_t><<<pgradgrid, pgradnthreads, 0, paraconv2dgradstream>>>(grad, p, input, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, height, width, channels, m, out_height, out_width, 1, pgrad);

// Compute the gradient for the extra dimension of the focal point.
  dim3 p0gradgrid(DIVCEIL(n,32)*32,DIVCEIL(out_height,32)*32,DIVCEIL(m,32)*32);
  int p0length = 8;
  dim3 p0gradnthreads(out_width,1,1);
  ParaConv2DGradP0KernelNCHW<scalar_t><<<p0gradgrid, p0gradnthreads, 0, paraconv2dgradstream>>>(grad, p0, n, m, out_height, out_width, 1, p0grad);

  d = height*width*channels;

// Apply a constant scaling to the gradients.
  
  blocksize = DIVCEIL(n,32)*32;
  int inputlength = 128;
  int inputnthreads = DIVCEIL(d, inputlength);
  ParaConv2DGrad2TensorCudaKernel<scalar_t><<<blocksize, inputnthreads, 0, paraconv2dgradstream>>>(n, d, inputlength, inputgrad);

  d = kernel_size0*kernel_size1*channels;
  
  blocksize = DIVCEIL(m,32)*32;
  int hlength = 128;
  int hnthreads = DIVCEIL(d, hlength);
  ParaConv2DGrad2FactorTensorCudaKernel<scalar_t><<<blocksize, hnthreads, 0, paraconv2dgradstream>>>(factor, m, d, hlength, hgrad);

  blocksize = DIVCEIL(m,32)*32;
  h0length = 128;
  int h0nthreads = DIVCEIL(1, h0length);
  ParaConv2DGrad2FactorTensorCudaKernel<scalar_t><<<blocksize, h0nthreads, 0, paraconv2dgradstream>>>(factor, m, 1, h0length, h0grad);

  blocksize = DIVCEIL(m,32)*32;
  int plength = 128;
  int pnthreads = DIVCEIL(d, plength);
  ParaConv2DGrad2FactorTensorCudaKernel<scalar_t><<<blocksize, pnthreads, 0, paraconv2dgradstream>>>(factor, m, d, plength, pgrad);

  blocksize = DIVCEIL(m,32)*32;
  p0length = 128;
  int p0nthreads = DIVCEIL(1, p0length);
  ParaConv2DGrad2FactorTensorCudaKernel<scalar_t><<<blocksize, p0nthreads, 0, paraconv2dgradstream>>>(factor, m, 1, p0length, p0grad);

  cudaStreamDestroy(paraconv2dgradstream);

}//paraconv2d_backward_cuda_main
template void paraconv2d_backward_cuda_main<float>(at::Device dev, const float* grad, const float* input, const float* parameters, const float* hsum, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, float* inputgrad, float* parametersgrad, bool skip_input_grad);
template void paraconv2d_backward_cuda_main<double>(at::Device dev, const double* grad, const double* input, const double* parameters, const double* hsum, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, double* inputgrad, double* parametersgrad, bool skip_input_grad);



