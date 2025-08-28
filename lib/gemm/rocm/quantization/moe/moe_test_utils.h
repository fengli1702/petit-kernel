// lib/gemm/rocm/quantization/moe/moe_test_utils.h
#pragma once

#include "moe_gemm_fp4.h"
#include "mxfp4_types.h"
#include <vector>
#include <tuple>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// 共用数据结构
// ============================================================================

/**
 * 量化FFN权重数据结构
 */
struct QuantizedFFNWeights {
    // W1权重和scales
    std::vector<unsigned> w1_data;
    std::vector<float> w1_scales;
    
    // W2权重和scales  
    std::vector<unsigned> w2_data;
    std::vector<float> w2_scales;
    
    // 元数据
    MxFp4WeightMetadata w1_metadata;
    MxFp4WeightMetadata w2_metadata;
};

// ============================================================================
// CPU参考量化实现
// ============================================================================

class CPUMxFp4Reference {
public:
    static constexpr float cpu_e2m1_lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,        // 正值 (0-7)
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // 负值 (8-15)
    };
    
    /**
     * 计算最优量化scale
     */
    static float CalculateOptimalScale(const std::vector<float>& block_values);
    
    /**
     * 量化为E2M1索引
     */
    static unsigned char QuantizeToE2M1Index(float value, float scale);
    
    /**
     * CPU E2M1反量化
     */
    static float CPUDequantE2M1(unsigned fp4_val, float scale);
};

// ============================================================================
// BF16转换辅助函数
// ============================================================================

namespace bf16_utils {
    /**
     * BF16到float转换（通过位操作）
     */
    inline float Bfloat16ToFloat(unsigned short bf16_val) {
        unsigned f32_val = static_cast<unsigned>(bf16_val) << 16;
        return reinterpret_cast<const float&>(f32_val);
    }
    
    /**
     * float到BF16转换（通过位操作，简单截断）
     */
    inline unsigned short FloatToBfloat16(float f_val) {
        unsigned f32_bits = reinterpret_cast<const unsigned&>(f_val);
        return static_cast<unsigned short>(f32_bits >> 16);
    }
}

// ============================================================================
// FP16数据准备函数
// ============================================================================

namespace fp16_data {
    /**
     * 准备完整的FFN测试数据（FP16）
     */
    std::tuple<std::vector<half>, QuantizedFFNWeights, std::vector<unsigned>> 
    PrepareFFNTestData(const MoEStage2Config& config, bool use_random_data = true);
    
    /**
     * 量化FFN权重（FP16版本）
     */
    QuantizedFFNWeights QuantizeFFNWeights(
        const std::vector<half>& w1_weights,
        const std::vector<half>& w2_weights, 
        const MoEStage2Config& config
    );
    
    /**
     * 单层权重量化（FP16版本）
     */
    void QuantizeWeightsLayer(
        const std::vector<half>& original_weights,
        std::vector<unsigned>& quantized_data,
        std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
    
    /**
     * CPU反量化权重（FP16版本）
     */
    std::vector<half> CPUDequantizeWeights(
        const std::vector<unsigned>& quantized_data,
        const std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
}

// ============================================================================
// BF16数据准备函数
// ============================================================================

namespace bf16_data {
    /**
     * 准备完整的FFN测试数据（BF16）
     */
    std::tuple<std::vector<unsigned short>, QuantizedFFNWeights, std::vector<unsigned>> 
    PrepareFFNTestData(const MoEStage2Config& config, bool use_random_data = true);
    
    /**
     * 量化FFN权重（BF16版本）
     */
    QuantizedFFNWeights QuantizeFFNWeights(
        const std::vector<unsigned short>& w1_weights,
        const std::vector<unsigned short>& w2_weights, 
        const MoEStage2Config& config
    );
    
    /**
     * 单层权重量化（BF16版本）
     */
    void QuantizeWeightsLayer(
        const std::vector<unsigned short>& original_weights,
        std::vector<unsigned>& quantized_data,
        std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
    
    /**
     * CPU反量化权重（BF16版本）
     */
    std::vector<unsigned short> CPUDequantizeWeights(
        const std::vector<unsigned>& quantized_data,
        const std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
}

// ============================================================================
// GPU反量化函数（已存在于kernel中）
// ============================================================================

namespace gpu_dequant {
    /**
     * GPU反量化权重（FP16版本）- 调用现有kernel
     */
    std::vector<half> GPUDequantizeWeightsFP16(
        const std::vector<unsigned>& quantized_data,
        const std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
    
    /**
     * GPU反量化权重（BF16版本）- 调用现有kernel
     */
    std::vector<unsigned short> GPUDequantizeWeightsBF16(
        const std::vector<unsigned>& quantized_data,
        const std::vector<float>& scales,
        unsigned input_dim, unsigned output_dim
    );
}

} // namespace causalflow::petit::rocm::quantization::moe