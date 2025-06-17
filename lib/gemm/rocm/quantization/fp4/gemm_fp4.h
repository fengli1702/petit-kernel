#pragma once

#include "gemm/rocm/quantization/types.h"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization::fp4 {

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n);

} // namespace causalflow::petit::rocm::quantization::fp4
