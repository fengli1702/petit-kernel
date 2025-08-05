#!/bin/sh

ROCM_ARCH="gfx90a;gfx942"
TORCH_PATH=$(pip show torch|grep Location|awk '{print $2;}')

CMAKE_ARGS="-DCMAKE_PREFIX_PATH=/opt/rocm;${TORCH_PATH} -DCMAKE_HIP_ARCHITECTURES=${ROCM_ARCH} -DGPU_TARGETS=${ROCM_ARCH}"

export CMAKE_ARGS
python3 setup.py clean --all
python3 setup.py bdist_wheel --dist-dir=dist
python3 -m auditwheel repair dist/petit_kernel-*-cp*-cp*-linux_x86_64.whl --exclude '*' -w dist/
