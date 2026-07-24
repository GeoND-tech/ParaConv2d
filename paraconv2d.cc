/*

Copyright 2026 Nikolaos Tsapanos, see the LICENSE file for specifics.

This file provides the interface between the python part of the library and the CUDA functions and kernels that carry out the computation.

The forward pass receives the input, the parameters of the operation and the parameters of the convolution and returns the output and hsum (hsum is a part of the output and will be needed again in the backward pass).

The backward pass receives the incoming gradient, the input, the hsum computed in the forward pass, the parameters of the operation and the parameters of the convolution and returns the relevant gradients.

NOTES

1) For the math involved, please refer to:
  - https://geond.tech/wp-content/uploads/2024/06/NPDBINNCP.pdf
  - https://geond.tech/wp-content/uploads/2026/07/PNDNN.pdf

2) In this implementation, the type scalar_t can be either float or double.

3) Keep in mind that PyTorch tensors are essentially vectors of contiguous data with metadata indexing.

  The pointer to the start of the vector is tensorname.data_ptr<scalar_t>() and its type is scalar_t*.

4) While this custom op has 4 parameters, h0, h, p and p0, they are all packed in a single vector in the listed order.

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

5) PyTorch passes floating point arguments as double precision, this is why we retrieve doublefactor as double and the cast it to (scalar_t).

*/

#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <ATen/ATen.h>
#include <vector>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#ifdef DIVCEIL
#undef DIVCEIL
#endif
#define DIVCEIL(x,y) ((x+y-1)/y)

/*
The name declared below is how the forward and backward functions are accessed from python.

In this case it's "torch.ops.ptparaconv2d", see the file ptparaconv2d.py for details.
*/
TORCH_LIBRARY(ptparaconv2d, m) {
  m.def("forward(Tensor input, Tensor parameters, float doublefactor, Tensor kernel_size_tensor, Tensor strides_tensor, Tensor dilation_rate_tensor) -> (Tensor[])");
  m.def("backward(Tensor grad, Tensor input, Tensor parameters, Tensor hsum, float doublefactor, Tensor kernel_size_tensor, Tensor strides_tensor, Tensor dilation_rate_tensor, bool skip_input_grad) -> (Tensor[])");
}



template <typename scalar_t>
void paraconv2d_forward_cpu_main(const scalar_t* input, const scalar_t* parameters, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* output, scalar_t* hsum) {

// This if the forward pass implementation for CPU.
    
    int d = channels*kernel_size0*kernel_size1;
    
    const scalar_t* h0 = parameters;
    const scalar_t* h = h0+(m);
    const scalar_t* p = h+(m*d);
    const scalar_t* p0 = p+(m*d);
    const scalar_t factor = (scalar_t)doublefactor;
    for (int i=0;i<n;i++) {
      for (int x1=0;x1<out_height;x1++) {
        for (int y1=0;y1<out_width;y1++) {
          for (int j=0;j<m;j++) {
            scalar_t tmp = h0[j];
            for (int k=0; k<channels; k++) {
              for (int ix2=0; ix2<kernel_size0; ix2++) {
                int x2 = (kernel_size0/2) - ix2;
                for (int iy2=0; iy2<kernel_size1; iy2++) {
                  int y2 = (kernel_size1/2) - iy2;
                  tmp += h[ j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] * input[i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))];
                }//for y2
              }//for x2
            }//for k
            hsum[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] = tmp;
            output[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] = tmp*tmp;
          }//for j
        }//for y1
      }//for x1
    }//for i

    for (int i=0;i<n;i++) {
      for (int x1=0;x1<out_height;x1++) {
        for (int y1=0;y1<out_width;y1++) {
          for (int j=0;j<m;j++) {
            for (int k=0; k<channels; k++) {
              for (int ix2=0; ix2<kernel_size0; ix2++) {
                int x2 = (kernel_size0/2) - ix2;
                for (int iy2=0; iy2<kernel_size1; iy2++) {
                  int y2 = (kernel_size1/2) - iy2;
                  scalar_t tmp = input[i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] - p[ j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2];
                  output[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1]-=tmp*tmp;
                }//for y2
              }//for x2
            }//for k
            output[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] -= p0[j]*p0[j];
            output[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1]*=factor;
          }//for j
        }//for y1
      }//for x1
    }//for i
    
}//paraconv2d_forward_cpu_main



template <typename scalar_t>
void paraconv2d_backward_cpu_main(const scalar_t* grad, const scalar_t* input, const scalar_t* parameters, const scalar_t* hsum, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* inputgrad, scalar_t* parametersgrad, bool skip_input_grad) {

// This if the backward pass implementation for CPU.

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

    std::memset(inputgrad, 0, n*height*width*channels*sizeof(scalar_t));
    std::memset(hgrad, 0, m*kernel_size0*kernel_size1*channels*sizeof(scalar_t));
    std::memset(h0grad, 0, m*sizeof(scalar_t));
    std::memset(pgrad, 0, m*kernel_size0*kernel_size1*channels*sizeof(scalar_t));
    std::memset(p0grad, 0, m*sizeof(scalar_t));

    if (!skip_input_grad) {
      for (int i=0;i<n;i++) {
        for (int x1=0;x1<out_height;x1++) {
          for (int y1=0;y1<out_width;y1++) {
            for (int k=0; k<channels; k++) {
              for (int ix2=0; ix2<kernel_size0; ix2++) {
                //int x2 = ix2 - kernel_size0/2;
                int x2 = (kernel_size0/2) - ix2;
                for (int iy2=0; iy2<kernel_size1; iy2++) {
                  //int y2 = iy2 - kernel_size1/2;
                  int y2 = (kernel_size1/2) - iy2;
                  scalar_t inputgradtmp = 0;
                  for (int j=0;j<m;j++) {
                    inputgradtmp += grad[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * ( ( hsum[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] *  h[j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2]) -  input[ i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] + p[j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] );
                  }//for j
                  inputgrad[i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] += inputgradtmp;
                }//for iy2
              }//for ix2
            }//for k
          }//for y1
        }//for x1
      }//for i
      for (int i=0;i<n;i++) {
        for (int j=0;j<height;j++) {
          for (int k=0;k<width;k++) {
            for (int l=0;l<channels;l++) {
                inputgrad[i*channels*height*width + l*height*width + j*width + k] *= 2;
            }//for l
          }//for k
        }//for j
      }//for i
    }//if !skip_input_grad

    for (int i=0;i<n;i++) {
      for (int x1=0;x1<out_height;x1++) {
        for (int y1=0;y1<out_width;y1++) {
          for (int j=0;j<m;j++) {
            //for (int x2=-kernel_size0/2; x2<=(kernel_size0/2)-(kernel_size0%2==0); x2++) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              //for (int y2=-kernel_size1/2; y2<=(kernel_size1/2-(kernel_size1%2==0)); y2++) {
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                for (int k=0; k<channels; k++) {
                  hgrad[j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] += grad[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * hsum[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * input[ i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))];
                }//for k
              }//for iy2
            }//for ix2
            h0grad[j] += grad[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * hsum[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1];
          }//for j
        }//for y1
      }//for x1
    }//for i

    for (int i=0;i<n;i++) {
      for (int x1=0;x1<out_height;x1++) {
        for (int y1=0;y1<out_width;y1++) {
          for (int j=0;j<m;j++) {
            for (int ix2=0; ix2<kernel_size0; ix2++) {
              int x2 = (kernel_size0/2) - ix2;
              for (int iy2=0; iy2<kernel_size1; iy2++) {
                int y2 = (kernel_size1/2) - iy2;
                for (int k=0; k<channels; k++) {
                  pgrad[j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2] += grad[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * ( input[ i*channels*height*width + k*height*width + ((x1*strides0)+((x2+(kernel_size0/2))*dilation_rate0))*width + ((y1*strides1)+((y2+(kernel_size1/2))*dilation_rate1))] - p[j*channels*kernel_size0*kernel_size1 + k*kernel_size0*kernel_size1 + ix2*kernel_size1 + iy2]);
                }//for k
              }//for iy2
            }//for ix2
            p0grad[j] += grad[i*m*out_height*out_width + j*out_height*out_width + x1*out_width + y1] * (-p0[j]);
          }//for j
        }//for y1
      }//for x1
    }//for i

    for (int i=0;i<m;i++) {
      h0grad[i] *= 2*factor;
      p0grad[i] *= 2*factor;
      for (int j=0;j<kernel_size0;j++) {
        for (int k=0;k<kernel_size1;k++) {
          for (int l=0;l<channels;l++) {
              hgrad[i*channels*kernel_size0*kernel_size1 + l*kernel_size0*kernel_size1 + j*kernel_size1 + k] *= 2*factor;
              pgrad[i*channels*kernel_size0*kernel_size1 + l*kernel_size0*kernel_size1 + j*kernel_size1 + k] *= 2*factor;
            }//for l
        }//for k
      }//for j
    }//for if

}//paraconv2d_backward_cpu_main



std::vector<torch::Tensor> paraconv2d_forward_cpu(
    torch::Tensor input,
    torch::Tensor parameters,
    double doublefactor, // python passes floats as double precision.
    torch::Tensor kernel_size_tensor,
    torch::Tensor strides_tensor,
    torch::Tensor dilation_rate_tensor
) {

// This function is the entry point for the CPU forward pass.

    int kernel_size0 = kernel_size_tensor.data_ptr<int>()[0];
    int kernel_size1 = kernel_size_tensor.data_ptr<int>()[1];
    int strides0 = strides_tensor.data_ptr<int>()[0];
    int strides1 = strides_tensor.data_ptr<int>()[1];
    int dilation_rate0 = dilation_rate_tensor.data_ptr<int>()[0];
    int dilation_rate1 = dilation_rate_tensor.data_ptr<int>()[1];

    int n = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int d = channels*kernel_size0*kernel_size1;
    int m = parameters.size(0)/(2*(d+1));

    int out_height = ((height-(dilation_rate0*(kernel_size0-1))-1)/strides0)+1;
    int out_width = ((width-(dilation_rate1*(kernel_size1-1))-1)/strides1)+1;
    
    auto output = torch::zeros({n, m, out_height, out_width}, input.scalar_type());
    auto hsum = torch::zeros({n, m, out_height, out_width}, input.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "paraconv2d_forward_cpu_main", ([&] {
      paraconv2d_forward_cpu_main<scalar_t>(input.data_ptr<scalar_t>(), parameters.data_ptr<scalar_t>(), doublefactor, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, channels, height, width, m, out_height, out_width, output.data_ptr<scalar_t>(), hsum.data_ptr<scalar_t>());
    }));

    return{output, hsum};

}//paraconv2d_forward_cpu



template <typename scalar_t>
void paraconv2d_forward_cuda_main(at::Device dev, const scalar_t* input, const scalar_t* parameters, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* output, scalar_t* hsum);



std::vector<torch::Tensor> paraconv2d_forward_cuda(
    torch::Tensor input,
    torch::Tensor parameters,
    double doublefactor, // python passes floats as double precision.
    torch::Tensor kernel_size_tensor,
    torch::Tensor strides_tensor,
    torch::Tensor dilation_rate_tensor
) {

// This function is the entry point for the CUDA forward pass.

    int kernel_size0 = kernel_size_tensor.data_ptr<int>()[0];
    int kernel_size1 = kernel_size_tensor.data_ptr<int>()[1];
    int strides0 = strides_tensor.data_ptr<int>()[0];
    int strides1 = strides_tensor.data_ptr<int>()[1];
    int dilation_rate0 = dilation_rate_tensor.data_ptr<int>()[0];
    int dilation_rate1 = dilation_rate_tensor.data_ptr<int>()[1];

    int n = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int d = channels*kernel_size0*kernel_size1;
    int m = parameters.size(0)/(2*(d+1));

    int out_height = ((height-(dilation_rate0*(kernel_size0-1))-1)/strides0)+1;
    int out_width = ((width-(dilation_rate1*(kernel_size1-1))-1)/strides1)+1;
    
    auto options = torch::TensorOptions().dtype(input.scalar_type()).device(input.device());
    
    auto output = torch::zeros({n, m, out_height, out_width}, options);
    auto hsum = torch::zeros({n, m, out_height, out_width}, options);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "paraconv2d_forward_cuda_main", ([&] {
      paraconv2d_forward_cuda_main<scalar_t>(input.device(), input.data_ptr<scalar_t>(), parameters.data_ptr<scalar_t>(), doublefactor, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, channels, height, width, m, out_height, out_width, output.data_ptr<scalar_t>(), hsum.data_ptr<scalar_t>());
    }));

    return{output, hsum};

}//paraconv2d_forward_cuda



std::vector<torch::Tensor> paraconv2d_backward_cpu(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor parameters,
    torch::Tensor hsum,
    double doublefactor, // python passes floats as double precision.
    torch::Tensor kernel_size_tensor,
    torch::Tensor strides_tensor,
    torch::Tensor dilation_rate_tensor,
    bool skip_input_grad
) {

// This function is the entry point for the CPU backward pass.

    int kernel_size0 = kernel_size_tensor.data_ptr<int>()[0];
    int kernel_size1 = kernel_size_tensor.data_ptr<int>()[1];
    int strides0 = strides_tensor.data_ptr<int>()[0];
    int strides1 = strides_tensor.data_ptr<int>()[1];
    int dilation_rate0 = dilation_rate_tensor.data_ptr<int>()[0];
    int dilation_rate1 = dilation_rate_tensor.data_ptr<int>()[1];



    int n = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int d = channels*kernel_size0*kernel_size1;
    int m = parameters.size(0)/(2*(d+1));

    int out_height = ((height-(dilation_rate0*(kernel_size0-1))-1)/strides0)+1;
    int out_width = ((width-(dilation_rate1*(kernel_size1-1))-1)/strides1)+1;

    auto inputgrad = torch::zeros({n, channels, height, width}, input.scalar_type());
    auto parametersgrad = torch::zeros({m * 2*(d+1)}, input.scalar_type());

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "paraconv2d_backward_cpu_main", ([&] {
      paraconv2d_backward_cpu_main<scalar_t>(grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), parameters.data_ptr<scalar_t>(), hsum.data_ptr<scalar_t>(), doublefactor, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, channels, height, width, m, out_height, out_width, inputgrad.data_ptr<scalar_t>(), parametersgrad.data_ptr<scalar_t>(), skip_input_grad);
    }));

    return{inputgrad, parametersgrad};

}//paraconv2d_backward_cpu



template <typename scalar_t>
void paraconv2d_backward_cuda_main(at::Device dev, const scalar_t* grad, const scalar_t* input, const scalar_t* parameters, const scalar_t* hsum, double doublefactor, const int kernel_size0, const int kernel_size1, const int strides0, const int strides1, const int dilation_rate0, const int dilation_rate1, int n, int channels, int height, int width, int m, int out_height, int out_width, scalar_t* inputgrad, scalar_t* parametersgrad, bool skip_input_grad);



std::vector<torch::Tensor> paraconv2d_backward_cuda(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor parameters,
    torch::Tensor hsum,
    double doublefactor, // python passes floats as double precision.
    torch::Tensor kernel_size_tensor,
    torch::Tensor strides_tensor,
    torch::Tensor dilation_rate_tensor,
    bool skip_input_grad
) {

// This function is the entry point for the CUDA backward pass.

    int kernel_size0 = kernel_size_tensor.data_ptr<int>()[0];
    int kernel_size1 = kernel_size_tensor.data_ptr<int>()[1];
    int strides0 = strides_tensor.data_ptr<int>()[0];
    int strides1 = strides_tensor.data_ptr<int>()[1];
    int dilation_rate0 = dilation_rate_tensor.data_ptr<int>()[0];
    int dilation_rate1 = dilation_rate_tensor.data_ptr<int>()[1];

    int n = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);
    int d = channels*kernel_size0*kernel_size1;
    int m = parameters.size(0)/(2*(d+1));

    int out_height = ((height-(dilation_rate0*(kernel_size0-1))-1)/strides0)+1;
    int out_width = ((width-(dilation_rate1*(kernel_size1-1))-1)/strides1)+1;

    auto options = torch::TensorOptions().dtype(input.scalar_type()).device(input.device());

    auto inputgrad = torch::zeros({n, channels, height, width}, options);
    auto parametersgrad = torch::zeros({m * 2*(d+1)}, options);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "paraconv2d_backward_cuda_main", ([&] {
      paraconv2d_backward_cuda_main<scalar_t>(input.device(), grad.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), parameters.data_ptr<scalar_t>(), hsum.data_ptr<scalar_t>(), doublefactor, kernel_size0, kernel_size1, strides0, strides1, dilation_rate0, dilation_rate1, n, channels, height, width, m, out_height, out_width, inputgrad.data_ptr<scalar_t>(), parametersgrad.data_ptr<scalar_t>(), skip_input_grad);
    }));

    return{inputgrad, parametersgrad};

}//paraconv2d_backward_cuda



TORCH_LIBRARY_IMPL(ptparaconv2d, CPU, m) {
  m.impl("forward", paraconv2d_forward_cpu);
}

TORCH_LIBRARY_IMPL(ptparaconv2d, CPU, m) {
  m.impl("backward", paraconv2d_backward_cpu);
}

TORCH_LIBRARY_IMPL(ptparaconv2d, CUDA, m) {
  m.impl("forward", paraconv2d_forward_cuda);
}

TORCH_LIBRARY_IMPL(ptparaconv2d, CUDA, m) {
  m.impl("backward", paraconv2d_backward_cuda);
}

