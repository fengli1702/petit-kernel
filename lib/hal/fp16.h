#pragma once

#include "causalflow/petit/config.h"

#ifdef WITH_CUDA
#include <cuda_fp16.h>
#else
#ifdef WITH_ROCM
#include <hip/hip_fp16.h>
#endif
#endif
