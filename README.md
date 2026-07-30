# ParaConv2d custom PyTorch op

This repository contains the source code for the paraboloid 2d convolution operation of the geondptfree PyTorch custom ops library. It is **NOT** an open source project, but a source-available project made public for educational purposes. You may clone and experiment with this repository, use it as a template to make your own custom op, but you cannot copy large portions of the code and/or use it for commercial purposes. See the LICENSE file for details.

## Implementation

This is a very early implementation of the operation and is the one included in the [free version](https://pypi.org/project/geondptfree/) of the GeoND library for pytorch. The [full version](https://geond.tech/licenses/) of the library includes a much more heavily optimized implementation that is 3.7 times faster for the forward pass and 17.6 times faster for the forward+backward pass, see [this repository](https://github.com/GeoND-tech/GeoND-examples).

## Requirements

- Linux
- Python >= 3.10
- CUDA toolkit >= 12.6
- PyTorch >= 2.9.0
- gcc/g++ that supports C++17 (C++20 for PyTorch 2.13.0)

## File structure

- paraconv2d.cc: Contains the interface with python and the CPU implementations.
- paraconv2d.cu: Launches the CUDA kernels that carry out the GPU computation.
- paraconv2dkernels.cu: Contains the code for the forward pass kernels.
- paraconv2dgradkernels.cu: Contains the code for the backward pass kernels.
- ptparaconv2s.py: Provides the PyTorch module ParaConv2d.

## Building

Instead of using CMAKE or PyTorch, this repository can be built using a custom script. This is more transparent and straight forward and also makes including other object files in the build simpler.

The script expects to find the GNU c++ compiler in /usr/bin/, CUDA's libcurand.so in /usr/lib/x86_64-linux-gnu/ and CUDA's runtime libcudart.so in /usr/local/lib/. If these files are elsewhere, either edit the script file or place these files in their expected location. This also allows you to control exactly which CUDA files the custom op is linked against.

To build, clone the repository, go inside the directory and run:

```sh
chmod +x buildpt.sh &&
./buildpt.sh paraconv2d
```

If your PyTorch version is earlier than 2.13.0, run:

```sh
chmod +x buildptpre2130.sh &&
./buildptpre2130.sh paraconv2d
```

These custom script files allow modification and redistribution as long as you don't remove the copyright notice inside them.

## Running an example

You can test the op by comparing it to the corresponding op of the geondptfree library as their outputs should match exactly:

```sh
python -m venv pt2130 &&
source pt2130/bin/activate &&
pip install torch==2.13.0 &&
chmod +x buildpt.sh &&
./buildpt.sh paraconv2d &&
pip install geondptfree==2.13.0.1.2rc2 &&
python ptparaconv2dtest.py
```

