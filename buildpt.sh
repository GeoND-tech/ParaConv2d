# Copyright 2026 Nikolaos Tsapanos. You may modify and redistribute this file as long as you do not remove this copyright notice.
# This is a build script for a PyTorch custom op C++/CUDA shared object file for PyTorch versions 2.13.0+. For versions earlier than 2.13 use buildptpre2130.sh instead.
# It uses the 6.0 compute capability architecture as a minimum, replace "-arch sm_60" with "-arch sm_XY" to build for X.Y.

set -e



# Start by retrieving some include and library directories.

TORCH_INCLUDE=( $(python -c 'import torch; import os; print(os.path.dirname(torch.__file__)+"/include")') )
TORCH_API_INCLUDE=( $(python -c 'import torch; import os; print(os.path.dirname(torch.__file__)+"/include/torch/csrc/api/include")') )
TORCH_LIB=( $(python -c 'import torch; import os; print(os.path.dirname(torch.__file__)+"/lib")') )
PYTHON_INCLUDE=( $(python -c 'import sysconfig; print(sysconfig.get_path("include"))') )



# Build the forward pass kernels object file.

nvcc -std=c++20 -v -c -o ./$1kernels.cu.o ./$1kernels.cu -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -arch sm_60 --device-c -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem $TORCH_INCLUDE -isystem $TORCH_API_INCLUDE -isystem $PYTHON_INCLUDE -D_GLIBCXX_USE_CXX11_ABI=1



# Build the backward pass kernels object file.

nvcc -std=c++20 -v -c -o ./$1gradkernels.cu.o ./$1gradkernels.cu -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -arch sm_60 --device-c -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem $TORCH_INCLUDE -isystem $TORCH_API_INCLUDE -D_GLIBCXX_USE_CXX11_ABI=1



# Build the CUDA computations object file.

nvcc -std=c++20 -v -c -o $1.cu.o $1.cu -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -arch sm_60 --device-c -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -isystem $TORCH_INCLUDE -isystem $TORCH_API_INCLUDE -isystem $PYTHON_INCLUDE -D_GLIBCXX_USE_CXX11_ABI=1



# Use --device-link so that CUDA object files built separately can be put together.

nvcc $1.cu.o $1kernels.cu.o $1gradkernels.cu.o --output-file $1link.o --device-link -arch sm_60 -Xcompiler -fPIC



# Build the python interface object file.

/usr/bin/c++ -std=c++20 -DUSE_C10D_GLOO -DUSE_C10D_NCCL -DUSE_DISTRIBUTED -DUSE_RPC -DUSE_TENSORPIPE -Dptparaboloid_EXPORTS -isystem $TORCH_INCLUDE -isystem $TORCH_API_INCLUDE -fPIC -isystem $PYTHON_INCLUDE -D_GLIBCXX_USE_CXX11_ABI=1 -o $1.cc.o -c $1.cc



# Put everything together in a shared object file and link it against the PyTorch binaries.

/usr/bin/c++ -shared -s -fPIC -Wl,-soname,libpt$1.so -o libpt$1.so $1.cc.o $1.cu.o $1kernels.cu.o $1gradkernels.cu.o $1link.o -Wl,-rpath,$TORCH_LIB:/usr/local/lib $TORCH_LIB/libtorch.so $TORCH_LIB/libc10.so -lcuda -lnvrtc -lnvToolsExt /usr/local/lib/libcudart.so $TORCH_LIB/libc10_cuda.so -Wl,--no-as-needed,"$TORCH_LIB/libtorch_cpu.so" -Wl,--as-needed -lpthread $TORCH_LIB/libc10_cuda.so $TORCH_LIB/libc10.so $TORCH_LIB/libtorch_python.so -lcufft /usr/lib/x86_64-linux-gnu/libcurand.so -lcublas -lcudnn -Wl,--no-as-needed,"$TORCH_LIB/libtorch.so" -Wl,--as-needed -lnvToolsExt /usr/local/lib/libcudart.so



# Clean up the object files.

rm *.o
