#pragma once

#include "mxfp4_types.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization {

/*
 * 官方MxFP4 E2M1格式反量化函数
 * 
 * E2M1映射表（与PyTorch官方实现一致）：
 * Index 0-7:  正值 [0.0, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
 * Index 8-15: 负值 [-0.0, -0.0625, -0.125, -0.25, -0.5, -1.0, -2.0, -4.0]
 */
template <typename ElementType>
__device__ __forceinline__ float SimpleMxFp4DequantE2M1(unsigned fp4_val, float scale) {
    // E2M1查找表
    constexpr float e2m1_lut[16] = {
        0.0f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,        // 正值
        -0.0f, -0.0625f, -0.125f, -0.25f, -0.5f, -1.0f, -2.0f, -4.0f  // 负值
    };
    
    // 确保fp4_val在0-15范围内
    fp4_val = fp4_val & 0xF;
    
    // 查表获取基础值并应用scale
    return e2m1_lut[fp4_val] * scale;
}

} // namespace causalflow::petit::rocm::quantization