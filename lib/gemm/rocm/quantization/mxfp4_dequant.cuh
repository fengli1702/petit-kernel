#pragma once

#include "mxfp4_types.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization {

/*
 * 根据OCP Microscaling Formats规范的E2M1映射表：
 * Index 0-7:  正值 [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0]
 * Index 8-15: 负值 [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
 */
template <typename ElementType>
__device__ __forceinline__ float SimpleMxFp4DequantE2M1(unsigned fp4_val, float scale) {
    constexpr float e2m1_lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,        // 正值 (0-7)
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // 负值 (8-15)
    };
    
    // 确保fp4_val在0-15范围内
    fp4_val = fp4_val & 0xF;
    
    // 查表获取基础值并应用scale
    return e2m1_lut[fp4_val] * scale;
}

} // namespace causalflow::petit::rocm::quantization