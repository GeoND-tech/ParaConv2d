import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
from torch.autograd import Variable, Function
import math
import sys
import itertools

torch.ops.load_library("libptparaconv2d.so")

def spotlightinit(m, d, input_factor, h_factor = 1.0, p_factor = -0.000001):
  h = torch.empty(m,d)
  h.data.uniform_(-input_factor*0.0001, input_factor*0.0001)
  h0 = h_factor*torch.ones(m)
  p = torch.empty(m,d)
  p.data.uniform_(-input_factor, input_factor)
  p0 = p_factor*torch.ones(m)
  return(h, p, h0, p0)



def pnnconvert(w, b, offset=20.0):
  H, D = w.size()
  tmpd=torch.sqrt(torch.sum(torch.square(w), dim=1))
  h = Variable(torch.div(w, tmpd.repeat(D,1).t()), requires_grad = False)
  tmpb = torch.div(b,tmpd)
  factor = 1.
  h0 = Variable(factor*(tmpb + offset), requires_grad = False)
  p = Variable(h.mul((factor*offset) - tmpb.repeat(D,1).t()), requires_grad = False)
  h = factor*h
  return(h, p, h0)



class ParaConv2dFunction(Function):
  @staticmethod
  def forward(ctx, input, pars, output_factor, kernel_size, stride, dilation, skip_input_grad, grad_factor):
    inputdtype = input.dtype
    if (inputdtype==torch.float16):
      input=input.float()
    output, hsum = torch.ops.ptparaconv2d.forward(input, pars, float(output_factor), kernel_size, stride, dilation)
    ctx.save_for_backward(input, pars, hsum)
    ctx.output_factor = output_factor
    ctx.kernel_size = kernel_size
    ctx.stride = stride
    ctx.dilation = dilation
    ctx.skip_input_grad = skip_input_grad
    ctx.grad_factor = grad_factor
    return output.to(inputdtype)
  @staticmethod
  def backward(ctx, grad_output):
    input, pars, hsum = ctx.saved_tensors
    graddtype = grad_output.dtype
    if (graddtype==torch.float16):
      grad_output=grad_output.float()    
    grad_input, grad_pars = torch.ops.ptparaconv2d.backward(grad_output, input, pars, hsum, float(ctx.output_factor), ctx.kernel_size, ctx.stride, ctx.dilation, ctx.skip_input_grad)
    return ctx.grad_factor*grad_input, grad_pars, None, None, None, None, None, None



class ParaConv2d(nn.Module):
  r"""
  Applies a 2D convolution over an input signal composed of several input planes using the paraboloid neuron computation.
  
  The arguments ``kernel_size``, ``stride``, ``padding``, ``dilation`` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension.
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension.
  
  This module currently does not support grouping.
  
  Args:
    in_channels
      Number of channels in the input image.
      
    out_channels
      Number of channels produced by the convolution.
      
    kernel_size
      Size of the convolving kernel.
      
    stride
      Stride of the convolution. Default: ``1``.
      
    padding
      Padding added to all four sides of the input. Default: ``0``.
    
    dilation
      Spacing between kernel elements. Default: ``1``.
      
    bias
      This is to facilitate ease of replacement of Linear layers with Paraboloid ones, does not do anything. Default: ``True``.
      
    padding_mode
      Same as torch.nn.functional.pad from PyTorch. Default: ``'constant'``.
    
    output_factor
      Multiplies the output of the module. Default: ``0.1``.

    input_factor
      Multiplies the input before passing it through the layer. Default: ``1.0``.

    lr_factor
      Multiplies the learning rate applied to the parameters by the optimizer. Default: ``100.0``.
    
    wd_factor
      Multiplies the weight decay applied to the parameters by the optimizer. Default: ``0.1``.
      
    skip_input_grad
      If set to ``True``, it skips the computation of the delta signal, should only be set for the very first layer of the network. Default: ``False``.
    
    init
      Selects the initialization method for the parameters. Valid options are ``'spotlight'``, ``'linear'``. Default: ``'spotlight'``.
      
    h_factor
      Affects the ``'spotlight'`` and ``'live'`` initializations. Multiplies the magnitude of the directrix vector. Default: ``0.01``.
      
    p_factor
      Affects the ``'spotlight'`` and ``'live'`` initializations. Determines the offset of the focus from the data subspace. Default: ``0.0001``.
      
    grad_factor
      Multiplies the outgoing delta signal. Default: ``1.0``.
      
    init_from_numpy
      Initiates the parameter tensor directly from a numpy tensor. Default: ``None``.      

----
      
  **Shape:**
  
    - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
    
    - Output: :math:`(N, C_{out}, H_{out}, W_{out})`, where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

----

  **Example:**

    >>> pb = gd.ParaConv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    >>> input = torch.randn(20, 16, 50, 100)
    >>> output = pb(input)


  """
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, padding_mode='constant', device = None, dtype = None, output_factor = 1.0, input_factor = 1.0, h_lr_factor = 1., p_lr_factor = 1., wd_factor = 0.1, skip_input_grad = False, init = 'spotlight', h_factor = 0.01, p_factor = 0.0001, grad_factor = 1., init_from_numpy = None):
    super(ParaConv2d, self).__init__()
    if (type(kernel_size) == int):
      kernel_size = [kernel_size, kernel_size]
    if (type(dilation) == int):
      dilation = [dilation, dilation]
    if (type(stride) == int):
      stride = [stride, stride]
    if (type(padding) == int):
      padding = (padding, padding, padding, padding)
    input_features = in_channels*kernel_size[0]*kernel_size[1]
    tmp = torch.empty(out_channels, input_features+1)
    stdv = 1. / math.sqrt(tmp.size(1))
    tmp.data.uniform_(-stdv, stdv)
    if init == 'spotlight':
      tmph, tmpp, tmph0, tmpp0 = spotlightinit(out_channels, input_features, input_factor, h_factor, p_factor)
    elif init == "linear":
      w = tmp[:, 0:input_features]
      b = tmp[:, input_features]
      tmph, tmpp, tmph0 = pnnconvert(w, b, 20.0)
      tmpp0 = 0.01*torch.ones(out_channels)
    else:
      raise Exception("Unknown Paraboloid initialization.")      
    self.in_channels = in_channels
    self.out_channels = out_channels
    if (dtype is not None):
      tmph0 = tmph0.to(dtype)
      tmph = tmph.to(dtype)
      tmpp0 = tmpp0.to(dtype)
      tmpp = tmpp.to(dtype)
    if (device is not None):
      tmph0 = tmph0.to(device)
      tmph = tmph.to(device)
      tmpp0 = tmpp0.to(device)
      tmpp = tmpp.to(device)
    if (init_from_numpy is not None):
      self.pars = nn.Parameter(torch.from_numpy(init_from_numpy))
    else:
      self.pars = nn.Parameter(torch.concat( (tmph0,torch.reshape(tmph,(-1,)),torch.reshape(tmpp,(-1,)),tmpp0), dim=0))
    self.pars.d = int(in_channels*torch.prod(torch.Tensor(kernel_size)))
    self.pars.isParaboloid = True
    self.pars.h_lr_factor = h_lr_factor
    self.pars.p_lr_factor = p_lr_factor
    self.pars.wd_factor = wd_factor
    self.output_factor = output_factor

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = torch.tensor(kernel_size, dtype = torch.int)
    self.stride = torch.tensor(stride, dtype = torch.int)
    self.dilation = torch.tensor(dilation, dtype = torch.int)
    self.skip_input_grad = skip_input_grad
    self.padding = padding
    self.padding_mode = padding_mode
    self.input_factor = input_factor
    self.grad_factor = grad_factor
    
    with torch.no_grad():
      self.forward(torch.zeros(1, in_channels, dilation[0]*kernel_size[0], dilation[1]*kernel_size[1])) # DO NOT REMOVE THIS LINE

  def forward(self, input):
    x = self.input_factor*input
    x = F.pad(x, self.padding, mode = self.padding_mode)
    return ParaConv2dFunction.apply(x, self.pars, self.output_factor, self.kernel_size, self.stride, self.dilation, self.skip_input_grad, self.grad_factor)

  def extra_repr(self) -> str:
    return f"in_channels={self.in_channels}, out_channels={self.out_channels}, input_factor={self.input_factor}, output_factor={self.output_factor}, kernel_size={self.kernel_size}"

