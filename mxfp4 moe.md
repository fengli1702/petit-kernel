# moe --mxfp4

## first step 

8.23

add [unit test](./lib/gemm/rocm/quantization/fp4/gemm_moe_fp4_fp16_rocm_test.cc) 

add [basic moe phase2 ](./lib/gemm/rocm/quantization/fp4/gemm_moe_fp4_fp16_grid.cc)

add [mxfp4](./lib/gemm/rocm/quantization/mxfp4_dequant.cuh) 

![unit test](./pic/unit%20test%208.23.png)

```bash
docker build -t petit-dev-rocm62 .

docker run -it \
  --name petit-dev-mi210 \
  --network=host \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  --ipc=host --shm-size=16g \
  --group-add=video --group-add=109 \
  --device /dev/kfd --device /dev/dri \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /home/o_feng:/home/o_feng \
  -v /nfs2:/nfs2 \
  -e HOME=/home/o_feng \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_haohui \
  -e CMAKE_ARGS='-DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx90a' \
  -e HIPCC_COMPILE_FLAGS_APPEND='-mllvm -amdgpu-load-store-vectorizer=0' \
  -e HIPCC_LINK_FLAGS_APPEND='-mllvm -amdgpu-load-store-vectorizer=0' \
  -u 10069 \
  petit-dev-rocm62 \
  /bin/bash


source ~/.venvs/petit/bin/activate

export CMAKE_HIP_ARCHITECTURES=gfx90a
export HIPCC_COMPILE_FLAGS_APPEND="-mllvm -amdgpu-load-store-vectorizer=0"
export HIPCC_LINK_FLAGS_APPEND="-mllvm -amdgpu-load-store-vectorizer=0"

rm -rf build/
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DWITH_ROCM=ON \
  -DWITH_PYTHON=OFF \
  -DCMAKE_PREFIX_PATH="/opt/rocm" \
  -DCMAKE_VERBOSE_MAKEFILE=ON \
  -GNinja

ninja -v

find . -name "*moe*" -type f

./lib/gemm/rocm/quantization/fp4/gemm_moe_fp4_fp16_rocm_test


```