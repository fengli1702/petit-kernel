#include "moe_test_utils.h"
#include "utils/hip_helper.h"
#include <climits>
#include <cmath>
#include <iostream>
#include <random>
#include <algorithm>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// CPUMxFp4Reference 实现
// ============================================================================

constexpr float CPUMxFp4Reference::cpu_e2m1_lut[16];

float CPUMxFp4Reference::CalculateOptimalScale(const std::vector<float>& block_values) {
    float max_abs = 0.0f;
    for (float val : block_values) {
        max_abs = std::max(max_abs, std::abs(val));
    }
    
    if (max_abs < 1e-9f) return 1.0f;
    
    float min_scale = max_abs / 6.0f;
    int exp_val = std::max(0, (int)std::ceil(std::log2f(min_scale)));
    return std::pow(2.0f, exp_val);
}

unsigned char CPUMxFp4Reference::QuantizeToE2M1Index(float value, float scale) {
    if (scale == 0.0f) scale = 1.0f;
    float scaled_value = value / scale;
    
    int best_idx = 0;
    float min_error = std::abs(scaled_value - cpu_e2m1_lut[0]);
    
    for (int i = 1; i < 16; i++) {
        float error = std::abs(scaled_value - cpu_e2m1_lut[i]);
        if (error < min_error) {
            min_error = error;
            best_idx = i;
        }
    }
    
    return static_cast<unsigned char>(best_idx);
}

float CPUMxFp4Reference::CPUDequantE2M1(unsigned fp4_val, float scale) {
    fp4_val = fp4_val & 0xF;
    return cpu_e2m1_lut[fp4_val] * scale;
}

// ============================================================================
// FP16数据准备函数实现
// ============================================================================

namespace fp16_data {

std::tuple<std::vector<half>, QuantizedFFNWeights, std::vector<unsigned>> 
PrepareFFNTestData(const MoEStage2Config& config, bool use_random_data) {
    std::vector<half> input_data(config.total_tokens * config.hidden_size);
    std::vector<half> w1_original(config.hidden_size * config.intermediate_size);
    std::vector<half> w2_original(config.intermediate_size * config.hidden_size);
    std::vector<unsigned> expert_indices(config.total_tokens, 0); // 单专家测试
    
    std::mt19937 gen(42);
    if (use_random_data) {
        std::normal_distribution<float> input_dist(0.0f, 0.3f);
        std::normal_distribution<float> weight_dist(0.0f, 0.2f);
        
        for (auto& val : input_data) val = __float2half(input_dist(gen));
        for (auto& val : w1_original) val = __float2half(weight_dist(gen));
        for (auto& val : w2_original) val = __float2half(weight_dist(gen));
    } else {
        // 使用更大的固定值，确保量化后不为零
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = __float2half(1.0f + 0.5f * (i % 10));  // 1.0~5.5
        }
        for (size_t i = 0; i < w1_original.size(); i++) {
            w1_original[i] = __float2half(0.5f + 0.2f * (i % 20)); // 0.5~4.3
        }
        for (size_t i = 0; i < w2_original.size(); i++) {
            w2_original[i] = __float2half(0.3f + 0.1f * (i % 15)); // 0.3~1.7
        }
    }
    
    // 量化W1和W2权重
    QuantizedFFNWeights ffn_weights = QuantizeFFNWeights(w1_original, w2_original, config);
    
    return std::make_tuple(input_data, ffn_weights, expert_indices);
}

QuantizedFFNWeights QuantizeFFNWeights(
    const std::vector<half>& w1_weights,
    const std::vector<half>& w2_weights, 
    const MoEStage2Config& config
) {
    QuantizedFFNWeights result;
    
    // 量化W1 (hidden_size -> intermediate_size)
    result.w1_data.resize((config.hidden_size * config.intermediate_size + 7) / 8, 0);
    result.w1_scales.resize((config.hidden_size / kMxFp4BlockSize) * config.intermediate_size);
    QuantizeWeightsLayer(w1_weights, result.w1_data, result.w1_scales, 
                       config.hidden_size, config.intermediate_size);
    
    // 量化W2 (intermediate_size -> hidden_size)
    result.w2_data.resize((config.intermediate_size * config.hidden_size + 7) / 8, 0);
    result.w2_scales.resize((config.intermediate_size / kMxFp4BlockSize) * config.hidden_size);
    QuantizeWeightsLayer(w2_weights, result.w2_data, result.w2_scales,
                       config.intermediate_size, config.hidden_size);
    
    // 设置元数据
    result.w1_metadata = MxFp4WeightMetadata::CalculateMetadata(
        config.hidden_size, config.intermediate_size, config.input_type);
    result.w2_metadata = MxFp4WeightMetadata::CalculateMetadata(
        config.intermediate_size, config.hidden_size, config.input_type);
    
    return result;
}

void QuantizeWeightsLayer(
    const std::vector<half>& original_weights,
    std::vector<unsigned>& quantized_data,
    std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    for (unsigned out_dim = 0; out_dim < output_dim; ++out_dim) {
        const unsigned packed_cols_stride = input_dim / 8;

        for (unsigned in_block = 0; in_block < input_dim / kMxFp4BlockSize; ++in_block) {
            std::vector<float> block_vals(kMxFp4BlockSize);
            for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                unsigned in_idx = in_block * kMxFp4BlockSize + i;
                unsigned weight_idx = in_idx * output_dim + out_dim;
                block_vals[i] = __half2float(original_weights[weight_idx]);
            }

            float scale = CPUMxFp4Reference::CalculateOptimalScale(block_vals);
            unsigned scale_idx = in_block * output_dim + out_dim;
            scales[scale_idx] = scale;

            for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                unsigned in_idx = in_block * kMxFp4BlockSize + i;
                float weight_val = block_vals[i];
                unsigned char fp4_idx = CPUMxFp4Reference::QuantizeToE2M1Index(weight_val, scale);
                unsigned packed_idx = out_dim * packed_cols_stride + (in_idx / 8);
                unsigned bit_offset = (in_idx % 8) * 4;
                quantized_data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
            }
        }
    }
}

std::vector<half> CPUDequantizeWeights(
    const std::vector<unsigned>& quantized_data,
    const std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    std::vector<half> dequant_weights(input_dim * output_dim);
    
    for (unsigned out_dim = 0; out_dim < output_dim; out_dim++) {
        const unsigned packed_cols_stride = input_dim / 8;
        
        for (unsigned in_dim = 0; in_dim < input_dim; in_dim++) {
            unsigned packed_idx = out_dim * packed_cols_stride + (in_dim / 8);
            unsigned bit_offset = (in_dim % 8) * 4;
            unsigned packed_weight = quantized_data[packed_idx];
            unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
            
            unsigned in_block = in_dim / kMxFp4BlockSize;
            unsigned scale_idx = in_block * output_dim + out_dim;
            float scale = scales[scale_idx];
            
            float weight_val = CPUMxFp4Reference::CPUDequantE2M1(fp4_val, scale);
            dequant_weights[in_dim * output_dim + out_dim] = __float2half(weight_val);
        }
    }
    
    return dequant_weights;
}

} // namespace fp16_data

// ============================================================================
// BF16数据准备函数实现
// ============================================================================

namespace bf16_data {

std::tuple<std::vector<unsigned short>, QuantizedFFNWeights, std::vector<unsigned>> 
PrepareFFNTestData(const MoEStage2Config& config, bool use_random_data) {
    std::vector<unsigned short> input_data(config.total_tokens * config.hidden_size);
    std::vector<unsigned short> w1_original(config.hidden_size * config.intermediate_size);
    std::vector<unsigned short> w2_original(config.intermediate_size * config.hidden_size);
    std::vector<unsigned> expert_indices(config.total_tokens, 0); // 单专家测试

    std::mt19937 gen(42);
    if (use_random_data) {
        std::normal_distribution<float> input_dist(0.0f, 0.3f);
        std::normal_distribution<float> weight_dist(0.0f, 0.2f);

        for (auto& val : input_data) val = bf16_utils::FloatToBfloat16(input_dist(gen));
        for (auto& val : w1_original) val = bf16_utils::FloatToBfloat16(weight_dist(gen));
        for (auto& val : w2_original) val = bf16_utils::FloatToBfloat16(weight_dist(gen));
    } else {
        // 使用与FP16非随机测试类似的大数值
        for (size_t i = 0; i < input_data.size(); i++) {
            input_data[i] = bf16_utils::FloatToBfloat16(1.0f + 0.5f * (i % 10));
        }
        for (size_t i = 0; i < w1_original.size(); i++) {
            w1_original[i] = bf16_utils::FloatToBfloat16(0.5f + 0.2f * (i % 20));
        }
        for (size_t i = 0; i < w2_original.size(); i++) {
            w2_original[i] = bf16_utils::FloatToBfloat16(0.3f + 0.1f * (i % 15));
        }
    }

    // 量化权重
    QuantizedFFNWeights ffn_weights = QuantizeFFNWeights(w1_original, w2_original, config);

    return std::make_tuple(input_data, ffn_weights, expert_indices);
}

QuantizedFFNWeights QuantizeFFNWeights(
    const std::vector<unsigned short>& w1_weights,
    const std::vector<unsigned short>& w2_weights, 
    const MoEStage2Config& config
) {
    QuantizedFFNWeights result;
    
    // 量化W1 (hidden_size -> intermediate_size)
    result.w1_data.resize((config.hidden_size * config.intermediate_size + 7) / 8, 0);
    result.w1_scales.resize((config.hidden_size / kMxFp4BlockSize) * config.intermediate_size);
    QuantizeWeightsLayer(w1_weights, result.w1_data, result.w1_scales, 
                       config.hidden_size, config.intermediate_size);
    
    // 量化W2 (intermediate_size -> hidden_size)
    result.w2_data.resize((config.intermediate_size * config.hidden_size + 7) / 8, 0);
    result.w2_scales.resize((config.intermediate_size / kMxFp4BlockSize) * config.hidden_size);
    QuantizeWeightsLayer(w2_weights, result.w2_data, result.w2_scales,
                       config.intermediate_size, config.hidden_size);
    
    // 设置元数据
    result.w1_metadata = MxFp4WeightMetadata::CalculateMetadata(
        config.hidden_size, config.intermediate_size, config.input_type);
    result.w2_metadata = MxFp4WeightMetadata::CalculateMetadata(
        config.intermediate_size, config.hidden_size, config.input_type);
    
    return result;
}

void QuantizeWeightsLayer(
    const std::vector<unsigned short>& original_weights,
    std::vector<unsigned>& quantized_data,
    std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    for (unsigned out_dim = 0; out_dim < output_dim; ++out_dim) {
        const unsigned packed_cols_stride = input_dim / 8;

        for (unsigned in_block = 0; in_block < input_dim / kMxFp4BlockSize; ++in_block) {
            std::vector<float> block_vals(kMxFp4BlockSize);
            for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                unsigned in_idx = in_block * kMxFp4BlockSize + i;
                unsigned weight_idx = in_idx * output_dim + out_dim;
                // 使用专家级BF16转换函数
                block_vals[i] = bf16_utils::Bfloat16ToFloat(original_weights[weight_idx]);
            }

            float scale = CPUMxFp4Reference::CalculateOptimalScale(block_vals);
            unsigned scale_idx = in_block * output_dim + out_dim;
            scales[scale_idx] = scale;

            for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                unsigned in_idx = in_block * kMxFp4BlockSize + i;
                float weight_val = block_vals[i];
                unsigned char fp4_idx = CPUMxFp4Reference::QuantizeToE2M1Index(weight_val, scale);
                unsigned packed_idx = out_dim * packed_cols_stride + (in_idx / 8);
                unsigned bit_offset = (in_idx % 8) * 4;
                quantized_data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
            }
        }
    }
}

std::vector<unsigned short> CPUDequantizeWeights(
    const std::vector<unsigned>& quantized_data,
    const std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    std::vector<unsigned short> dequant_weights(input_dim * output_dim);
    
    for (unsigned out_dim = 0; out_dim < output_dim; out_dim++) {
        const unsigned packed_cols_stride = input_dim / 8;
        
        for (unsigned in_dim = 0; in_dim < input_dim; in_dim++) {
            unsigned packed_idx = out_dim * packed_cols_stride + (in_dim / 8);
            unsigned bit_offset = (in_dim % 8) * 4;
            unsigned packed_weight = quantized_data[packed_idx];
            unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
            
            unsigned in_block = in_dim / kMxFp4BlockSize;
            unsigned scale_idx = in_block * output_dim + out_dim;
            float scale = scales[scale_idx];
            
            float weight_val = CPUMxFp4Reference::CPUDequantE2M1(fp4_val, scale);
            dequant_weights[in_dim * output_dim + out_dim] = bf16_utils::FloatToBfloat16(weight_val);
        }
    }
    
    return dequant_weights;
}

} // namespace bf16_data

// ============================================================================
// GPU反量化函数实现
// ============================================================================

namespace gpu_dequant {

std::vector<half> GPUDequantizeWeightsFP16(
    const std::vector<unsigned>& quantized_data,
    const std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    void *d_quant_weights, *d_scales, *d_dequant_weights;

    CheckHIPStatus(hipMalloc(&d_quant_weights, quantized_data.size() * sizeof(unsigned)));
    CheckHIPStatus(hipMalloc(&d_scales, scales.size() * sizeof(float)));
    CheckHIPStatus(hipMalloc(&d_dequant_weights, input_dim * output_dim * sizeof(half)));

    CheckHIPStatus(hipMemcpy(d_quant_weights, quantized_data.data(), 
                            quantized_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_scales, scales.data(), 
                            scales.size() * sizeof(float), hipMemcpyHostToDevice));

    // 调用 GPU 反量化 kernel
    int err = CallFullDequantMxFp4Kernel(
        d_dequant_weights,
        d_quant_weights,
        d_scales,
        input_dim,
        output_dim,
        DataType::kDataTypeFp16, nullptr
    );

    if (err != 0) {
        std::cerr << "CallFullDequantMxFp4Kernel failed with error: " << err << std::endl;
        CheckHIPStatus(hipFree(d_quant_weights));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_dequant_weights));
        return std::vector<half>();
    }
    
    CheckHIPStatus(hipDeviceSynchronize());

    std::vector<half> result(input_dim * output_dim);
    CheckHIPStatus(hipMemcpy(result.data(), d_dequant_weights, 
                            result.size() * sizeof(half), hipMemcpyDeviceToHost));

    CheckHIPStatus(hipFree(d_quant_weights));
    CheckHIPStatus(hipFree(d_scales));
    CheckHIPStatus(hipFree(d_dequant_weights));

    return result;
}

std::vector<unsigned short> GPUDequantizeWeightsBF16(
    const std::vector<unsigned>& quantized_data,
    const std::vector<float>& scales,
    unsigned input_dim, unsigned output_dim
) {
    void *d_quant_weights, *d_scales, *d_dequant_weights;

    CheckHIPStatus(hipMalloc(&d_quant_weights, quantized_data.size() * sizeof(unsigned)));
    CheckHIPStatus(hipMalloc(&d_scales, scales.size() * sizeof(float)));
    CheckHIPStatus(hipMalloc(&d_dequant_weights, input_dim * output_dim * sizeof(unsigned short)));

    CheckHIPStatus(hipMemcpy(d_quant_weights, quantized_data.data(), 
                            quantized_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_scales, scales.data(), 
                            scales.size() * sizeof(float), hipMemcpyHostToDevice));

    // 调用 GPU 反量化 kernel
    int err = CallFullDequantMxFp4Kernel(
        d_dequant_weights,
        d_quant_weights,
        d_scales,
        input_dim,
        output_dim,
        DataType::kDataTypeBf16, nullptr
    );

    if (err != 0) {
        std::cerr << "CallFullDequantMxFp4Kernel for BF16 failed with error: " << err << std::endl;
        CheckHIPStatus(hipFree(d_quant_weights));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_dequant_weights));
        return std::vector<unsigned short>();
    }
    
    CheckHIPStatus(hipDeviceSynchronize());

    std::vector<unsigned short> result(input_dim * output_dim);
    CheckHIPStatus(hipMemcpy(result.data(), d_dequant_weights, 
                            result.size() * sizeof(unsigned short), hipMemcpyDeviceToHost));

    CheckHIPStatus(hipFree(d_quant_weights));
    CheckHIPStatus(hipFree(d_scales));
    CheckHIPStatus(hipFree(d_dequant_weights));

    return result;
}

} // namespace gpu_dequant

} // namespace causalflow::petit::rocm::quantization::moe
