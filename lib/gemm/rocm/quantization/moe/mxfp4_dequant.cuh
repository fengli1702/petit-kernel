// lib/gemm/rocm/quantization/moe/mxfp4_dequant.cuh
#pragma once

#include "mxfp4_types.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <type_traits>

namespace causalflow::petit::rocm::quantization {

// ============================================================================
// GPU设备端MxFP4 E2M1反量化实现
// ============================================================================

/**
 * MxFP4 E2M1格式的查找表
 * 
 * 根据OCP Microscaling Formats规范的E2M1映射表：
 * Index 0-7:  正值 [+0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0]
 * Index 8-15: 负值 [-0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
 */
__device__ __constant__ static const float kE2M1LookupTable[16] = {
    // 正值部分 (index 0-7)
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,        
    // 负值部分 (index 8-15)  
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  
};

/**
 * GPU设备端MxFP4 E2M1反量化函数
 * 
 * @param fp4_val   4-bit量化值 (0-15)
 * @param scale     对应的缩放因子
 * @return          反量化后的float值
 */
template <typename ElementType>
__device__ __forceinline__ float SimpleMxFp4DequantE2M1(unsigned fp4_val, float scale) {
    // 确保fp4_val在0-15范围内
    fp4_val = fp4_val & 0xF;
    
    // 查表获取基础值并应用scale
    return kE2M1LookupTable[fp4_val] * scale;
}

// ============================================================================
// 针对不同数据类型的特化实现
// ============================================================================

/**
 * 针对half类型的优化反量化
 */
template <>
__device__ __forceinline__ float SimpleMxFp4DequantE2M1<half>(unsigned fp4_val, float scale) {
    fp4_val = fp4_val & 0xF;
    return kE2M1LookupTable[fp4_val] * scale;
}

/**
 * 针对__hip_bfloat16类型的优化反量化  
 */
template <>
__device__ __forceinline__ float SimpleMxFp4DequantE2M1<__hip_bfloat16>(unsigned fp4_val, float scale) {
    fp4_val = fp4_val & 0xF;
    return kE2M1LookupTable[fp4_val] * scale;
}

// ============================================================================
// 辅助工具函数
// ============================================================================

/**
 * 验证4-bit值是否在有效范围内
 */
__device__ __forceinline__ bool IsValidFp4Value(unsigned fp4_val) {
    return fp4_val <= 15;
}

/**
 * 获取E2M1格式的最大绝对值
 */
__device__ __forceinline__ float GetMaxAbsValue() {
    return kMxFp4E2M1MaxValue;  // 6.0f
}

/**
 * 计算给定值的理论最优量化scale
 * (主要用于调试和验证)
 */
__device__ __forceinline__ float CalculateOptimalScale(float max_abs_value) {
    if (max_abs_value < 1e-9f) return 1.0f;
    
    float min_scale = max_abs_value / GetMaxAbsValue();
    
    // 找到大于等于min_scale的最小2的幂次
    int exp_val = __float2int_ru(__log2f(min_scale));
    return __powf(2.0f, static_cast<float>(exp_val));
}

} // namespace causalflow::petit::rocm::quantization