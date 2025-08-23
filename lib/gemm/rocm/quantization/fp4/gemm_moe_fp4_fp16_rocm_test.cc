#include "gemm_moe_fp4.h"
#include "../mxfp4_types.h"
#include "tests/quantization.h"
#include "utils/hip_helper.h"
#include "gemm/rocm/quantization/gemm.h"

#include <climits>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include <bitset>

namespace causalflow::petit::rocm::quantization::fp4 {

// Helper to check hipBLASLt status
static inline void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLASLt Error: " << status << std::endl;
        throw std::runtime_error("HipBLASLt Error");
    }
}

// ============= 精度匹配器 =============

MATCHER_P2(IsNearFp16, ref, abs_tolerance, "") {
    // arg 和 ref 都是 __half 类型
    const float a_f = __half2float(arg);
    const float b_f = __half2float(ref);
    const float abs_diff = std::abs(a_f - b_f);

    // 1. 检查绝对误差
    if (abs_diff < abs_tolerance) {
        return true;
    }

    // 2. 检查相对误差 (避免除零)
    if (b_f != 0.0f) {
        const float rel_error = abs_diff / std::abs(b_f);
        if (rel_error < 0.01f) {  // 1% 相对误差容忍度
            return true;
        }
    }

    // 3. 分析FP16二进制表示 (1 sign, 5 exponent, 10 mantissa)
    const uint16_t arg_bits = *reinterpret_cast<const uint16_t*>(&arg);
    const uint16_t ref_bits = *reinterpret_cast<const uint16_t*>(&ref);

    const uint16_t arg_exp = (arg_bits >> 10) & 0x1F;
    const uint16_t ref_exp = (ref_bits >> 10) & 0x1F;

    // 如果符号和指数位完全相同，则检查尾数差异 (ULP)
    // 0xFC00 是 FP16 的符号位和指数位的掩码
    if ((arg_bits & 0xFC00) == (ref_bits & 0xFC00)) {
        const uint16_t arg_mantissa = arg_bits & 0x3FF;
        const uint16_t ref_mantissa = ref_bits & 0x3FF;
        if (std::abs(static_cast<int>(arg_mantissa) - static_cast<int>(ref_mantissa)) <= 4) { // 容忍4个ULP的误差
            return true;
        }
    }

    // 4. 指数位差1的边界情况
    if (std::abs(static_cast<int>(arg_exp) - static_cast<int>(ref_exp)) == 1 && b_f != 0.0f) {
        if ((abs_diff / std::abs(b_f)) < 0.05f) { // 放宽至5%相对误差
            return true;
        }
    }
    if (result_listener->IsInterested()) {
        *result_listener << "Expected fp16 value near " << b_f << " (0x" << std::hex << ref_bits
                         << "), but got " << a_f << " (0x" << std::hex << arg_bits << ")";
    }

    return false;
}

MATCHER_P2(IsNearBf16, ref, abs_tolerance, "") {
    // arg 和 ref 都是 hip_bfloat16 或类似类型
    const uint32_t a_u32 = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&arg)) << 16;
    const uint32_t b_u32 = static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&ref)) << 16;
    const float a_f = *reinterpret_cast<const float*>(&a_u32);
    const float b_f = *reinterpret_cast<const float*>(&b_u32);

    const float abs_diff = std::abs(a_f - b_f);

    // 1. 绝对误差检查
    if (abs_diff < abs_tolerance) {
        return true;
    }

    // 2. 相对误差检查
    if (b_f != 0.0f) {
        const float rel_error = abs_diff / std::abs(b_f);
        if (rel_error < 0.01f) { // 1% 相对误差容忍度
            return true;
        }
    }

    // 3. 分析BF16二进制表示 (1 sign, 8 exponent, 7 mantissa)
    const uint16_t arg_bits = *reinterpret_cast<const uint16_t*>(&arg);
    const uint16_t ref_bits = *reinterpret_cast<const uint16_t*>(&ref);

    // 如果符号和指数位完全相同，则检查尾数差异 (ULP)
    // 0xFF80 是 BF16 的符号位和指数位的掩码
    if ((arg_bits & 0xFF80) == (ref_bits & 0xFF80)) {
        const uint16_t arg_mantissa = arg_bits & 0x7F;
        const uint16_t ref_mantissa = ref_bits & 0x7F;
        if (std::abs(static_cast<int>(arg_mantissa) - static_cast<int>(ref_mantissa)) <= 2) { // BF16尾数更短, 容忍2个ULP
            return true;
        }
    }
    
    if (result_listener->IsInterested()) {
        *result_listener << "Expected bfloat16 value near " << b_f << " (0x" << std::hex << ref_bits
                         << "), but got " << a_f << " (0x" << std::hex << arg_bits << ")";
    }

    return false;
}
// 如果bfloat16转换函数不可用，提供辅助函数
inline float bfloat16_to_float(hip_bfloat16 bf) {
    uint16_t bits = *reinterpret_cast<uint16_t*>(&bf);
    uint32_t float_bits = static_cast<uint32_t>(bits) << 16;
    return *reinterpret_cast<float*>(&float_bits);
}

inline hip_bfloat16 float_to_bfloat16(float f) {
    uint32_t float_bits = *reinterpret_cast<uint32_t*>(&f);
    uint16_t bf16_bits = static_cast<uint16_t>(float_bits >> 16);
    return *reinterpret_cast<hip_bfloat16*>(&bf16_bits);
}

// ============= MxFP4量化器 =============

class OfficialMxFp4Quantizer {
public:
    static constexpr float e2m1_values[16] = {
        0.0f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
        -0.0f, -0.0625f, -0.125f, -0.25f, -0.5f, -1.0f, -2.0f, -4.0f
    };
    
    static float CalculateOptimalScale(const std::vector<float>& block_values) {
        float max_abs = 0.0f;
        for (float val : block_values) {
            max_abs = std::max(max_abs, std::abs(val));
        }
        
        if (max_abs < 1e-9f) {
            return 1.0f;
        }
        
        // 让最大值映射到2.0-3.0范围，保留一些余量
        float target_max = 2.5f;
        int exp_val = std::ceil(std::log2f(max_abs / target_max));
        return std::pow(2.0f, exp_val);
    }
    
    static unsigned char QuantizeToE2M1Index(float value, float scale) {
        if (scale == 0.0f) scale = 1.0f;
        float scaled_value = value / scale;
        
        int best_idx = 0;
        float min_error = std::abs(scaled_value - e2m1_values[0]);
        
        for (int i = 1; i < 16; i++) {
            float error = std::abs(scaled_value - e2m1_values[i]);
            if (error < min_error) {
                min_error = error;
                best_idx = i;
            }
        }
        
        return static_cast<unsigned char>(best_idx);
    }
    
    static float Dequantize(unsigned char index, float scale) {
        return e2m1_values[index & 0xF] * scale;
    }
};

constexpr float OfficialMxFp4Quantizer::e2m1_values[16];

// ============= 测试类 =============

class MoeMxFp4Test : public ::testing::Test {
public:
    void SetUp() override {
        CheckHIPStatus(hipSetDevice(0));
        CheckHIPStatus(hipMalloc(&d_global_scale_, sizeof(float)));
        
        // hipBLASLt setup
        CheckHipblasStatus(hipblasLtCreate(&hipblaslt_handle_));
        CheckHipblasStatus(hipblasLtMatmulDescCreate(&matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));

        auto plat = hal::GetPlatform("rocm");
        ASSERT_EQ(absl::OkStatus(), plat->GetDevice(0, &dev_));
    }

    void TearDown() override {
        CheckHIPStatus(hipFree(d_global_scale_));
        CheckHipblasStatus(hipblasLtMatmulDescDestroy(matmul_desc_));
        CheckHipblasStatus(hipblasLtDestroy(hipblaslt_handle_));
        CheckHIPStatus(hipDeviceSynchronize());
    }

protected:
    float *d_global_scale_;
    std::unique_ptr<hal::Device> dev_;
    hipblasLtHandle_t hipblaslt_handle_;
    hipblasLtMatmulDesc_t matmul_desc_;

    struct QuantizedWeights {
        std::vector<unsigned> data;
        std::vector<float> scales;
    };

    // 量化权重
    QuantizedWeights QuantizeWeights(
        const std::vector<half>& original_weights,
        unsigned hidden_size,
        unsigned intermediate_size
    ) {
        QuantizedWeights result;
        result.data.resize((hidden_size * intermediate_size + 7) / 8, 0);
        result.scales.resize((hidden_size / 32) * intermediate_size);

        for (unsigned out_dim = 0; out_dim < intermediate_size; ++out_dim) {
            const unsigned packed_cols_stride = hidden_size / 8;

            for (unsigned h_block = 0; h_block < hidden_size / 32; ++h_block) {
                std::vector<float> block_vals(32);
                for (unsigned i = 0; i < 32; ++i) {
                    unsigned h = h_block * 32 + i;
                    unsigned weight_idx = h * intermediate_size + out_dim;
                    block_vals[i] = __half2float(original_weights[weight_idx]);
                }

                float scale = OfficialMxFp4Quantizer::CalculateOptimalScale(block_vals);
                unsigned scale_idx = h_block * intermediate_size + out_dim;
                result.scales[scale_idx] = scale;

                for (unsigned i = 0; i < 32; ++i) {
                    unsigned h = h_block * 32 + i;
                    float weight_val = block_vals[i];
                    unsigned char fp4_idx = OfficialMxFp4Quantizer::QuantizeToE2M1Index(weight_val, scale);
                    unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
                    unsigned bit_offset = (h % 8) * 4;
                    result.data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
                }
            }
        }
        return result;
    }

    std::vector<half> DequantizeWeightsCPU(
        const QuantizedWeights& quantized,
        unsigned hidden_size,
        unsigned intermediate_size
    ) {
        std::vector<half> result(hidden_size * intermediate_size);
        const unsigned packed_cols_stride = hidden_size / 8;

        for (unsigned out_dim = 0; out_dim < intermediate_size; out_dim++) {
            for (unsigned h = 0; h < hidden_size; h++) {
                unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
                unsigned bit_offset = (h % 8) * 4;
                unsigned fp4_idx = (quantized.data[packed_idx] >> bit_offset) & 0xF;
                
                unsigned h_block = h / 32;
                unsigned scale_idx = h_block * intermediate_size + out_dim;
                float scale = quantized.scales[scale_idx];

                float weight_val = OfficialMxFp4Quantizer::Dequantize(fp4_idx, scale);
                result[h * intermediate_size + out_dim] = __float2half(weight_val);
            }
        }
        return result;
    }
    //计算量化后再反量化输入到hipBLAS中 
    std::vector<half> ComputeConsistencyReferenceHipBLAS(
        const std::vector<half>& input_data,
        const QuantizedWeights& quantized,
        float global_scale,
        unsigned m, unsigned n, unsigned k
    ) {
        std::cout << "  Dequantizing weights on CPU for consistency reference..." << std::endl;
        std::vector<half> dequantized_weights = DequantizeWeightsCPU(quantized, k, n);

        std::cout << "  Running hipBLASLt on dequantized weights..." << std::endl;
        // Reuse the existing hipBLASLt function with the dequantized weights
        return ComputeGroundTruthHipBLAS(input_data, dequantized_weights, global_scale, m, n, k);
    }
    // GPU计算
    std::vector<half> ComputeMxFp4GPU(
        const std::vector<half>& input_data,
        const std::vector<unsigned>& quantized_weights,
        const std::vector<float>& scales,
        const std::vector<unsigned>& expert_indices,
        float global_scale,
        unsigned total_tokens,
        unsigned hidden_size,
        unsigned intermediate_size
    ) {
        void *d_input, *d_weights, *d_indices, *d_scales, *d_output;
        
        CheckHIPStatus(hipMalloc(&d_input, input_data.size() * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_weights, quantized_weights.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_indices, expert_indices.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_scales, scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_output, total_tokens * intermediate_size * sizeof(half)));

        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), 
                                input_data.size() * sizeof(half), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_weights, quantized_weights.data(), 
                                quantized_weights.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_indices, expert_indices.data(), 
                                expert_indices.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_scales, scales.data(), 
                                scales.size() * sizeof(float), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_global_scale_, &global_scale, sizeof(float), hipMemcpyHostToDevice));

        PetitSolutionHints hints;
        hints.a_type = DataType::kDataTypeFp16;
        hints.b_type = DataType::kDataTypeFp4e2m1;
        hints.c_type = DataType::kDataTypeFp16;

        int result = MoeMxFp4SecondStage(
            reinterpret_cast<unsigned*>(d_output),
            reinterpret_cast<const unsigned*>(d_input),
            reinterpret_cast<const unsigned*>(d_weights),
            reinterpret_cast<const unsigned*>(d_indices),
            reinterpret_cast<const float*>(d_scales),
            d_global_scale_,
            total_tokens, hidden_size, intermediate_size,
            hints, nullptr
        );
        
        EXPECT_EQ(result, 0);
        CheckHIPStatus(hipDeviceSynchronize());

        std::vector<half> gpu_result(total_tokens * intermediate_size);
        CheckHIPStatus(hipMemcpy(gpu_result.data(), d_output, 
                                gpu_result.size() * sizeof(half), hipMemcpyDeviceToHost));

        CheckHIPStatus(hipFree(d_input));
        CheckHIPStatus(hipFree(d_weights));
        CheckHIPStatus(hipFree(d_indices));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_output));

        return gpu_result;
    }

    // CPU计算（量化版本）
    std::vector<half> ComputeQuantizedCPU(
        const std::vector<half>& input_data,
        const QuantizedWeights& quantized,
        float global_scale,
        unsigned total_tokens,
        unsigned hidden_size,
        unsigned intermediate_size
    ) {
        std::vector<half> result(total_tokens * intermediate_size);
        
        for (unsigned token = 0; token < total_tokens; token++) {
            for (unsigned out_dim = 0; out_dim < intermediate_size; out_dim++) {
                float accumulator = 0.0f;
                const unsigned packed_cols_stride = hidden_size / 8;
                
                for (unsigned h = 0; h < hidden_size; h++) {
                    float input_val = __half2float(input_data[token * hidden_size + h]);
                    unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
                    unsigned bit_offset = (h % 8) * 4;
                    unsigned fp4_idx = (quantized.data[packed_idx] >> bit_offset) & 0xF;
                    unsigned h_block = h / 32;
                    unsigned scale_idx = h_block * intermediate_size + out_dim;
                    float scale = quantized.scales[scale_idx];
                    float weight_val = OfficialMxFp4Quantizer::Dequantize(fp4_idx, scale);
                    accumulator += input_val * weight_val;
                }
                
                accumulator *= global_scale;
                result[token * intermediate_size + out_dim] = __float2half(accumulator);
            }
        }
        return result;
    }

    // CPU Ground Truth（原始权重）
    std::vector<half> ComputeGroundTruthCPU(
        const std::vector<half>& input_data,
        const std::vector<half>& original_weights,
        float global_scale,
        unsigned m, unsigned n, unsigned k
    ) {
        std::vector<half> result(m * n);
        
        for (unsigned mi = 0; mi < m; mi++) {
            for (unsigned ni = 0; ni < n; ni++) {
                float accumulator = 0.0f;
                for (unsigned ki = 0; ki < k; ki++) {
                    float input_val = __half2float(input_data[mi * k + ki]);
                    float weight_val = __half2float(original_weights[ki * n + ni]);
                    accumulator += input_val * weight_val;
                }
                accumulator *= global_scale;
                result[mi * n + ni] = __float2half(accumulator);
            }
        }
        return result;
    }

    // hipBLASLt Ground Truth
    std::vector<half> ComputeGroundTruthHipBLAS(
        const std::vector<half>& input_data,
        const std::vector<half>& original_weights,
        float global_scale,
        unsigned m, unsigned n, unsigned k
    ) {
        void *d_input, *d_weights, *d_output;
        CheckHIPStatus(hipMalloc(&d_input, m * k * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_weights, k * n * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_output, m * n * sizeof(half)));

        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), m * k * sizeof(half), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_weights, original_weights.data(), k * n * sizeof(half), hipMemcpyHostToDevice));

        // 创建矩阵布局
        hipblasLtMatrixLayout_t layout_a, layout_b, layout_c;
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16F, k, m, k));
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16F, n, k, n));
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_16F, n, m, n));

        float alpha = global_scale, beta = 0.0f;
        
        // C = alpha * B^T * A^T + beta * C
        CheckHipblasStatus(hipblasLtMatmul(
            hipblaslt_handle_, matmul_desc_, &alpha,
            d_weights, layout_b,
            d_input, layout_a,
            &beta,
            d_output, layout_c,
            d_output, layout_c,
            nullptr, nullptr, 0, nullptr
        ));
        
        CheckHIPStatus(hipDeviceSynchronize());

        std::vector<half> result(m * n);
        CheckHIPStatus(hipMemcpy(result.data(), d_output, m * n * sizeof(half), hipMemcpyDeviceToHost));

        CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_a));
        CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_b));
        CheckHipblasStatus(hipblasLtMatrixLayoutDestroy(layout_c));
        CheckHIPStatus(hipFree(d_input));
        CheckHIPStatus(hipFree(d_weights));
        CheckHIPStatus(hipFree(d_output));

        return result;
    }

    // 量化质量评估
    struct QuantizationMetrics {
        float avg_abs_error;
        float max_abs_error;
        float avg_rel_error;
        float max_rel_error;
        int sign_flips;
        int large_errors;  // 错误 > 0.5
        
        void Print() const {
            std::cout << "  Avg absolute error: " << avg_abs_error << std::endl;
            std::cout << "  Max absolute error: " << max_abs_error << std::endl;
            std::cout << "  Avg relative error: " << (avg_rel_error * 100) << "%" << std::endl;
            std::cout << "  Max relative error: " << (max_rel_error * 100) << "%" << std::endl;
            std::cout << "  Sign flips: " << sign_flips << std::endl;
            std::cout << "  Large errors (>0.5): " << large_errors << std::endl;
        }
        
        bool IsAcceptable() const {
            // 4-bit量化的合理阈值
            return avg_abs_error < 0.2f &&     // 平均误差 < 0.2
                   max_rel_error < 5.0f &&      // 最大相对误差 < 500%
                   sign_flips < 10;               // 符号翻转少于10个
        }
    };

    QuantizationMetrics AnalyzeError(
        const std::vector<half>& reference,
        const std::vector<half>& test,
        const std::string& name
    ) {
        QuantizationMetrics metrics = {0};
        size_t count = 0;
        
        for (size_t i = 0; i < reference.size(); i++) {
            float ref_val = __half2float(reference[i]);
            float test_val = __half2float(test[i]);
            float abs_error = std::abs(ref_val - test_val);
            
            metrics.avg_abs_error += abs_error;
            metrics.max_abs_error = std::max(metrics.max_abs_error, abs_error);
            
            if (std::abs(ref_val) > 1e-3f) {
                float rel_error = abs_error / std::abs(ref_val);
                metrics.avg_rel_error += rel_error;
                metrics.max_rel_error = std::max(metrics.max_rel_error, rel_error);
                count++;
            }
            
            if ((ref_val > 0 && test_val < 0) || (ref_val < 0 && test_val > 0)) {
                metrics.sign_flips++;
            }
            
            if (abs_error > 0.5f) {
                metrics.large_errors++;
            }
        }
        
        metrics.avg_abs_error /= reference.size();
        if (count > 0) {
            metrics.avg_rel_error /= count;
        }
        
        std::cout << "\n" << name << " Metrics:" << std::endl;
        metrics.Print();
        
        return metrics;
    }

    // 主测试函数
    void TestCorrectness(
        unsigned total_tokens,
        unsigned hidden_size,
        unsigned intermediate_size,
        float global_scale,
        bool use_random_data = false,
        bool test_implementation_consistency = false
    ) {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Testing: " << total_tokens << "x" << hidden_size 
                  << "x" << intermediate_size << " (scale=" << global_scale 
                  << ", random=" << use_random_data << ")" << std::endl;
        if (test_implementation_consistency) {
            std::cout << "Mode: Backend Implementation Consistency" << std::endl;
        } else {
            std::cout << "Mode: End-to-End Quantization Error" << std::endl;
        }
        std::cout << "======================================" << std::endl;


        // --- 数据生成 ---
        std::vector<half> input_data(total_tokens * hidden_size);
        std::vector<half> original_weights(hidden_size * intermediate_size);
        std::vector<unsigned> expert_indices(total_tokens, 0);

        std::mt19937 gen(42);
        if (use_random_data) {
            std::normal_distribution<float> input_dist(0.0f, 0.3f);
            std::normal_distribution<float> weight_dist(0.0f, 0.2f);
            for (auto& val : input_data) val = __float2half(input_dist(gen));
            for (auto& val : original_weights) val = __float2half(weight_dist(gen));
        } else {
            for (size_t i = 0; i < input_data.size(); i++) {
                input_data[i] = __float2half(0.1f + 0.01f * (i % 10));
            }
            for (size_t i = 0; i < original_weights.size(); i++) {
                original_weights[i] = __float2half(0.05f + 0.01f * (i % 20));
            }
        }

        // --- 步骤 1: 计算基准 (Ground Truth) - 逻辑分支 ---
        std::vector<half> gt_hipblas;
        if (test_implementation_consistency) {
            std::cout << "\n--- Step 1: Computing Backend Consistency Reference ---" << std::endl;
            std::cout << "  Quantizing weights first..." << std::endl;
            auto quantized = QuantizeWeights(original_weights, hidden_size, intermediate_size);
            std::cout << "  Dequantizing weights on CPU for reference..." << std::endl;
            auto dequantized_weights = DequantizeWeightsCPU(quantized, hidden_size, intermediate_size);
            std::cout << "  Running hipBLASLt on DEQUANTIZED weights..." << std::endl;
            gt_hipblas = ComputeGroundTruthHipBLAS(
                input_data, dequantized_weights, global_scale,
                total_tokens, intermediate_size, hidden_size);
        } else {
            std::cout << "\n--- Step 1: Computing Ground Truth (Pristine Weights) ---" << std::endl;
            auto gt_cpu = ComputeGroundTruthCPU(
                input_data, original_weights, global_scale,
                total_tokens, intermediate_size, hidden_size);
            
            gt_hipblas = ComputeGroundTruthHipBLAS(
                input_data, original_weights, global_scale,
                total_tokens, intermediate_size, hidden_size);

            std::cout << "Verifying CPU vs hipBLASLt consistency..." << std::endl;
            for (size_t i = 0; i < gt_cpu.size(); i++) {
                EXPECT_THAT(gt_cpu[i], IsNearFp16(gt_hipblas[i], 1e-3f));
            }
            auto gt_metrics = AnalyzeError(gt_hipblas, gt_cpu, "CPU vs hipBLASLt");
            EXPECT_LT(gt_metrics.max_abs_error, 1e-3f);
        }

        // 2. 量化权重
        std::cout << "\n--- Step 2: Quantizing Weights ---" << std::endl;
        auto quantized = QuantizeWeights(original_weights, hidden_size, intermediate_size);
        std::cout << "Quantized data size: " << quantized.data.size() << " uint32s" << std::endl;
        std::cout << "Scale count: " << quantized.scales.size() << std::endl;

        // 3. CPU量化计算
        std::cout << "\n--- Step 3: CPU Quantized Computation ---" << std::endl;
        auto cpu_quantized = ComputeQuantizedCPU(
            input_data, quantized, global_scale,
            total_tokens, hidden_size, intermediate_size);

        // 4. GPU量化计算
        std::cout << "\n--- Step 4: GPU Quantized Computation ---" << std::endl;
        auto gpu_result = ComputeMxFp4GPU(
            input_data, quantized.data, quantized.scales, expert_indices,
            global_scale, total_tokens, hidden_size, intermediate_size);

        // 5. 误差分析
        std::cout << "\n--- Step 5: Error Analysis ---" << std::endl;
        
        // 实现一致性（CPU量化 vs GPU量化）
        auto impl_metrics = AnalyzeError(cpu_quantized, gpu_result, "Implementation Consistency (CPU vs GPU)");
        
        // 6. 详细对比前10个元素
        std::cout << "\n--- Detailed Comparison (first 10) ---" << std::endl;
        std::cout << std::setw(5) << "Idx" 
                  << std::setw(12) << "GT(hipBLAS)" 
                  << std::setw(12) << "CPU Quant" 
                  << std::setw(12) << "GPU Quant" 
                  << std::setw(10) << "Q-Error" 
                  << std::setw(10) << "Impl-Err" << std::endl;
        std::cout << std::string(71, '-') << std::endl;
        
        for (size_t i = 0; i < std::min<size_t>(10, gpu_result.size()); i++) {
            float gt_val = __half2float(gt_hipblas[i]);
            float cpu_val = __half2float(cpu_quantized[i]);
            float gpu_val = __half2float(gpu_result[i]);
            
            std::cout << std::setw(5) << i 
                      << std::fixed << std::setprecision(6)
                      << std::setw(12) << gt_val
                      << std::setw(12) << cpu_val
                      << std::setw(12) << gpu_val
                      << std::setw(10) << std::abs(gt_val - gpu_val)
                      << std::setw(10) << std::abs(cpu_val - gpu_val);
            
            if (std::abs(cpu_val - gpu_val) > 1e-3f) {
                std::cout << " [!]";
            }
            std::cout << std::endl;
        }

        // 7. 最终验证
        std::cout << "\n--- Final Validation ---" << std::endl;

        EXPECT_LT(impl_metrics.max_abs_error, 1e-3f) 
            << "CPU and GPU quantized implementations should ALWAYS match closely";

        if (test_implementation_consistency) {
            // 模式2: 期望GPU结果与“一致性基准”非常接近
            for (size_t i = 0; i < gpu_result.size(); i++) {
                EXPECT_THAT(gpu_result[i], IsNearFp16(gt_hipblas[i], 1e-2f))
                    << "GPU result should be very close to consistency reference at index " << i;
            }
            if (::testing::Test::HasFailure()) {
                std::cout << "\n❌ TEST FAILED! (Backend Consistency Check)" << std::endl;
            } else {
                std::cout << "\n✅ TEST PASSED! (Backend Consistency Check)" << std::endl;
            }
        } else {
            // 模式1: 期望GPU结果在“量化误差”的可接受范围内
            // 在这里计算和使用量化误差指标
            auto quant_metrics_gpu = AnalyzeError(gt_hipblas, gpu_result, "Quantization Error (GT vs GPU Quantized)");

            unsigned total_output_elements = total_tokens * intermediate_size;
            int acceptable_sign_flips = std::max(10, (int)(total_output_elements * 0.1));

            bool quantization_is_acceptable = 
                quant_metrics_gpu.avg_abs_error < 0.2f &&
                quant_metrics_gpu.sign_flips < acceptable_sign_flips;

            EXPECT_TRUE(quantization_is_acceptable)
                << "Quantization error metrics should be within acceptable dynamic range.";

            for (size_t i = 0; i < gpu_result.size(); i++) {
                const float base_tolerance = 0.5f;
                float dynamic_tolerance = base_tolerance * (hidden_size / 32.0f);
                EXPECT_THAT(gpu_result[i], IsNearFp16(gt_hipblas[i], dynamic_tolerance * global_scale))
                    << "GPU result should be within quantization tolerance of GT at index " << i;
            }

            if (::testing::Test::HasFailure()) {
                 std::cout << "\n❌ TEST FAILED! (Quantization Error Check)" << std::endl;
            } else {
                 std::cout << "\n✅ TEST PASSED! (Quantization Error Check)" << std::endl;
            }
        }
    }

    // 专门测试E2M1值的精确表示
    void TestE2M1ExactValues() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Testing E2M1 Exact Value Representation" << std::endl;
        std::cout << "======================================" << std::endl;
 
        // E2M1可精确表示的值
        std::vector<float> exact_values = {
            0.0f, 0.0625f, 0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f,
            -0.0625f, -0.125f, -0.25f, -0.5f, -1.0f, -2.0f, -4.0f
        };
 
        std::cout << "\nVerifying exact E2M1 values can be perfectly quantized:" << std::endl;
        
        for (float test_val : exact_values) {
            // 创建一个全是该值的块
            std::vector<float> block(32, test_val);
            float scale = OfficialMxFp4Quantizer::CalculateOptimalScale(block);
            
            unsigned char idx = OfficialMxFp4Quantizer::QuantizeToE2M1Index(test_val, scale);
            float dequant = OfficialMxFp4Quantizer::Dequantize(idx, scale);
            
            float error = std::abs(test_val - dequant);
            std::cout << "  Value " << std::setw(8) << test_val 
                      << " -> scale=" << std::setw(8) << scale
                      << " -> idx=" << std::setw(2) << (int)idx
                      << " -> dequant=" << std::setw(8) << dequant
                      << " (error=" << error << ")";
            
            if (error < 1e-6f) {
                std::cout << " ✓" << std::endl;
            } else {
                std::cout << " ✗ ERROR!" << std::endl;
            }
            
            EXPECT_NEAR(test_val, dequant, 1e-6f) 
                << "E2M1 exact value should be perfectly representable";
        }
    }
 
    // 测试边界条件
    void TestBoundaryConditions() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Testing Boundary Conditions" << std::endl;
        std::cout << "======================================" << std::endl;
 
        struct TestCase {
            std::string name;
            std::vector<float> values;
        };
 
        std::vector<TestCase> test_cases = {
            {"All zeros", std::vector<float>(32, 0.0f)},
            {"All ones", std::vector<float>(32, 1.0f)},
            {"Very small values", std::vector<float>{1e-4f, 2e-4f, 3e-4f, 4e-4f}},
            {"Very large values", std::vector<float>{100.0f, 200.0f, 300.0f, 400.0f}},
            {"Mixed signs", std::vector<float>{-1.0f, 1.0f, -2.0f, 2.0f}},
            {"Near zero", std::vector<float>{0.001f, -0.001f, 0.01f, -0.01f}}
        };
 
        for (const auto& test_case : test_cases) {
            std::cout << "\nTesting: " << test_case.name << std::endl;
            
            // 扩展到32个值
            std::vector<float> block(32);
            for (size_t i = 0; i < 32; i++) {
                block[i] = test_case.values[i % test_case.values.size()];
            }
            
            float scale = OfficialMxFp4Quantizer::CalculateOptimalScale(block);
            std::cout << "  Optimal scale: " << scale << std::endl;
            
            // 量化和反量化
            float total_error = 0.0f;
            float max_error = 0.0f;
            int sign_flips = 0;
            
            for (float val : block) {
                unsigned char idx = OfficialMxFp4Quantizer::QuantizeToE2M1Index(val, scale);
                float dequant = OfficialMxFp4Quantizer::Dequantize(idx, scale);
                float error = std::abs(val - dequant);
                
                total_error += error;
                max_error = std::max(max_error, error);
                
                if ((val > 0 && dequant < 0) || (val < 0 && dequant > 0)) {
                    sign_flips++;
                }
             }
             
             std::cout << "  Average error: " << (total_error / block.size()) << std::endl;
             std::cout << "  Maximum error: " << max_error << std::endl;
             std::cout << "  Sign flips: " << sign_flips << std::endl;
         }
     }
};

// ============= 测试用例 =============
/*-----------------------------------绝对精度损失测试--------------------------------------------*/

//// 基础正确性测试
//TEST_F(MoeMxFp4Test, BasicCorrectness) {
//   TestCorrectness(1, 32, 8, 1.0f, false);
//}
//
//// E2M1精确值测试
//TEST_F(MoeMxFp4Test, E2M1ExactValues) {
//   TestE2M1ExactValues();
//}
//
//// 边界条件测试
//TEST_F(MoeMxFp4Test, BoundaryConditions) {
//   TestBoundaryConditions();
//}
//
//// 小规模随机数据
//TEST_F(MoeMxFp4Test, RandomDataSmall) {
//   TestCorrectness(2, 32, 16, 1.0f, true);
//}
//
//// 中等规模随机数据
//TEST_F(MoeMxFp4Test, RandomDataMedium) {
//   TestCorrectness(4, 64, 32, 1.0f, true);
//}
//
//// 大规模随机数据
//TEST_F(MoeMxFp4Test, RandomDataLarge) {
//   TestCorrectness(8, 128, 64, 1.0f, true);
//}

//// 不同的global scale测试
//TEST_F(MoeMxFp4Test, DifferentScales) {
//   std::vector<float> scales = {0.5f, 1.0f, 2.0f, 4.0f};
//   for (float scale : scales) {
//       TestCorrectness(2, 32, 16, scale, true);
//   }
//}

//// 压力测试
//TEST_F(MoeMxFp4Test, StressTest) {
//   // 逐步增加规模
//   std::vector<std::tuple<unsigned, unsigned, unsigned>> configs = {
//       {4, 64, 64},
//       {8, 128, 128},
//       {16, 256, 256}
//   };
//   
//   for (const auto& [tokens, hidden, inter] : configs) {
//       std::cout << "\n=== Stress Test: " << tokens << "x" << hidden << "x" << inter << " ===" << std::endl;
//       TestCorrectness(tokens, hidden, inter, 1.0f, true);
//   }
//}


/*-----------------------------------准确性测试--------------------------------------------*/

// 基础正确性测试
TEST_F(MoeMxFp4Test, BasicCorrectness) {
   TestCorrectness(1, 32, 8, 1.0f, false, true);
}

// E2M1精确值测试
TEST_F(MoeMxFp4Test, E2M1ExactValues) {
   TestE2M1ExactValues();
}

// 边界条件测试
TEST_F(MoeMxFp4Test, BoundaryConditions) {
   TestBoundaryConditions();
}

// 小规模随机数据
TEST_F(MoeMxFp4Test, RandomDataSmall) {
   TestCorrectness(2, 32, 16, 1.0f, true, true);
}

// 中等规模随机数据
TEST_F(MoeMxFp4Test, RandomDataMedium) {
   TestCorrectness(4, 64, 32, 1.0f, true, true);
}

// 大规模随机数据
TEST_F(MoeMxFp4Test, RandomDataLarge) {
   TestCorrectness(8, 128, 64, 1.0f, true, true);
}
// 不同的global scale测试
TEST_F(MoeMxFp4Test, DifferentScales) {
   std::vector<float> scales = {0.5f, 1.0f, 2.0f, 4.0f};
   for (float scale : scales) {
       TestCorrectness(2, 32, 16, scale, true, true);
   }
}
// 压力测试
TEST_F(MoeMxFp4Test, StressTest) {
   // 逐步增加规模
   std::vector<std::tuple<unsigned, unsigned, unsigned>> configs = {
       {4, 64, 64},
       {8, 128, 128},
       {16, 256, 256}
   };
   
   for (const auto& [tokens, hidden, inter] : configs) {
       std::cout << "\n=== Stress Test: " << tokens << "x" << hidden << "x" << inter << " ===" << std::endl;
       TestCorrectness(tokens, hidden, inter, 1.0f, true, true);
   }
}
} // namespace causalflow::petit::rocm::quantization::fp4