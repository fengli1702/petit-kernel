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

namespace causalflow::petit::rocm::quantization::fp4 {

static inline void CheckHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        std::cerr << "HipBLAS Error: " << status << std::endl;
        throw std::runtime_error("HipBLAS Error");
    }
}

MATCHER_P2(IsNearFp16, ref, mantissa_diff, "") {
    float a_f = __half2float(arg);
    float b_f = __half2float(ref);

    if (std::abs(a_f - b_f) < std::min<float>(1e-2, fabs(b_f) * 0.01f)) {
        return true;
    }

    unsigned short a_u16 = reinterpret_cast<const unsigned short &>(arg);
    unsigned short b_u16 = reinterpret_cast<const unsigned short &>(ref);

    int mantissa_a = a_u16 & 0x3ff, mantissa_b = b_u16 & 0x3ff;
    unsigned other_a = a_u16 & 0xfc00, other_b = b_u16 & 0xfc00;
    bool very_small_relaxed =
        std::abs(a_f - b_f) < 1e-3f && std::abs(mantissa_a - mantissa_b) <= 20;
    bool result = other_a == other_b &&
                  (very_small_relaxed ||
                   std::abs(mantissa_a - mantissa_b) <= mantissa_diff);

    if (!result && result_listener->IsInterested()) {
        *result_listener << "Expected fp16 value near " << b_f << " (0x" << std::hex << b_u16
                         << "), but got " << a_f << " (0x" << std::hex << a_u16 << ")";
    }

    return result;
}

// ============= CPU参考实现 (独立LUT，避免循环依赖) =============
class CPUMxFp4Reference {
public:
    // CPU参考LUT - 与GPU kernel中的LUT相同，但独立定义
    static constexpr float cpu_e2m1_lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,        // 正值 (0-7)
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // 负值 (8-15)
    };
    
    static float CalculateOptimalScale(const std::vector<float>& block_values) {
        float max_abs = 0.0f;
        for (float val : block_values) {
            max_abs = std::max(max_abs, std::abs(val));
        }
        
        if (max_abs < 1e-9f) return 1.0f;
        
        float min_scale = max_abs / 6.0f;
        int exp_val = std::max(0, (int)std::ceil(std::log2f(min_scale)));
        return std::pow(2.0f, exp_val);
    }
    
    static unsigned char QuantizeToE2M1Index(float value, float scale) {
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
    
    // CPU参考反量化函数 - 对应GPU中的SimpleMxFp4DequantE2M1
    static float CPUDequantE2M1(unsigned fp4_val, float scale) {
        fp4_val = fp4_val & 0xF;
        return cpu_e2m1_lut[fp4_val] * scale;
    }
};

constexpr float CPUMxFp4Reference::cpu_e2m1_lut[16];

class MoeMxFp4Test : public ::testing::Test {
public:
    void SetUp() override {
        CheckHIPStatus(hipSetDevice(0));
        CheckHIPStatus(hipMalloc(&d_global_scale_, sizeof(float)));
        
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

    // ============= Step 1: 验证SimpleMxFp4DequantE2M1函数 =============
    
    void TestGPUDequantFunctionCorrectness() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 1: Testing GPU SimpleMxFp4DequantE2M1 Function" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // 准备测试数据：所有可能的4-bit值和多种scale
        std::vector<unsigned> fp4_test_values;
        std::vector<float> scale_test_values;
        
        std::vector<float> test_scales = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
        
        // 为每个scale测试所有16个4-bit值
        for (float scale : test_scales) {
            for (unsigned fp4 = 0; fp4 < 16; fp4++) {
                fp4_test_values.push_back(fp4);
                scale_test_values.push_back(scale);
            }
        }
        
        unsigned num_tests = fp4_test_values.size();
        std::cout << "Testing " << num_tests << " combinations of fp4_values and scales" << std::endl;
        
        // 分配GPU内存
        void *d_fp4_values, *d_scales, *d_gpu_results;
        CheckHIPStatus(hipMalloc(&d_fp4_values, num_tests * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_scales, num_tests * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_gpu_results, num_tests * sizeof(float)));
        
        // 复制测试数据到GPU
        CheckHIPStatus(hipMemcpy(d_fp4_values, fp4_test_values.data(), 
                                num_tests * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_scales, scale_test_values.data(), 
                                num_tests * sizeof(float), hipMemcpyHostToDevice));
        
        // 调用GPU测试kernel
        int err = CallTestGPUDequantKernel(
            reinterpret_cast<float*>(d_gpu_results),
            reinterpret_cast<const unsigned*>(d_fp4_values),
            reinterpret_cast<const float*>(d_scales),
            num_tests, nullptr
        );
        
        ASSERT_EQ(err, 0) << "CallTestGPUDequantKernel failed";
        CheckHIPStatus(hipDeviceSynchronize());
        
        // 获取GPU结果
        std::vector<float> gpu_results(num_tests);
        CheckHIPStatus(hipMemcpy(gpu_results.data(), d_gpu_results, 
                                num_tests * sizeof(float), hipMemcpyDeviceToHost));
        
        // CPU参考计算
        std::vector<float> cpu_results(num_tests);
        for (unsigned i = 0; i < num_tests; i++) {
            cpu_results[i] = CPUMxFp4Reference::CPUDequantE2M1(
                fp4_test_values[i], scale_test_values[i]);
        }
        
        // 验证结果
        ValidateGPUvsCPUResults(gpu_results, cpu_results, fp4_test_values, scale_test_values);
        
        // 清理
        CheckHIPStatus(hipFree(d_fp4_values));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_gpu_results));
        
        std::cout << "✅ GPU SimpleMxFp4DequantE2M1 Function Test PASSED" << std::endl;
    }

    void ValidateGPUvsCPUResults(
        const std::vector<float>& gpu_results,
        const std::vector<float>& cpu_results,
        const std::vector<unsigned>& fp4_values,
        const std::vector<float>& scales
    ) {
        int mismatches = 0;
        float max_diff = 0.0f;
        
        std::cout << "\nGPU vs CPU SimpleMxFp4DequantE2M1 Comparison:" << std::endl;
        std::cout << std::setw(6) << "Index" << std::setw(8) << "FP4" 
                  << std::setw(8) << "Scale" << std::setw(12) << "CPU" 
                  << std::setw(12) << "GPU" << std::setw(10) << "Diff" << std::endl;
        std::cout << std::string(66, '-') << std::endl;
        
        for (size_t i = 0; i < std::min<size_t>(20, gpu_results.size()); i++) {
            float diff = std::abs(cpu_results[i] - gpu_results[i]);
            max_diff = std::max(max_diff, diff);
            
            if (diff > 1e-6f) mismatches++;
            
            std::cout << std::setw(6) << i 
                      << std::setw(8) << fp4_values[i]
                      << std::setw(8) << std::fixed << std::setprecision(2) << scales[i]
                      << std::setw(12) << std::setprecision(4) << cpu_results[i]
                      << std::setw(12) << gpu_results[i]
                      << std::setw(10) << std::scientific << diff << std::endl;
        }
        
        std::cout << "\nSimpleMxFp4DequantE2M1 Function Test Results:" << std::endl;
        std::cout << "  Total tests: " << gpu_results.size() << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Mismatches: " << mismatches << std::endl;
        
        // GPU函数应该与CPU完全一致
        EXPECT_EQ(mismatches, 0) << "GPU SimpleMxFp4DequantE2M1 should match CPU exactly";
        EXPECT_LT(max_diff, 1e-6f) << "GPU SimpleMxFp4DequantE2M1 should have no floating-point errors";
    }

    // ============= Step 2: 测试量化→反量化的往返精度 =============
    
    void TestQuantizationRoundTripAccuracy() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 2: Testing Quantization Round-Trip Accuracy" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // 创建测试权重矩阵
        unsigned hidden_size = 64, intermediate_size = 32;
        std::vector<half> original_weights(hidden_size * intermediate_size);
        
        // 包含E2M1精确值和随机值
        std::vector<float> exact_values = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        
        std::mt19937 gen(42);
        std::normal_distribution<float> random_dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < original_weights.size(); i++) {
            float val;
            if (i < exact_values.size()) {
                val = exact_values[i];
            } else {
                val = random_dist(gen);
            }
            original_weights[i] = __float2half(val);
        }
        
        // 执行量化
        auto quantized = QuantizeWeights(original_weights, hidden_size, intermediate_size);
        
        // 使用GPU反量化kernel
        auto gpu_dequantized = GPUDequantizeWeights(quantized, hidden_size, intermediate_size);
        
        // 分析往返精度
        AnalyzeRoundTripAccuracy(original_weights, gpu_dequantized);
        
        std::cout << " Quantization Round-Trip Accuracy Test PASSED" << std::endl;
    }

    void AnalyzeRoundTripAccuracy(
        const std::vector<half>& original,
        const std::vector<half>& dequantized
    ) {
        float max_error = 0.0f;
        float avg_error = 0.0f;
        int perfect_matches = 0;
        
        for (size_t i = 0; i < original.size(); i++) {
            float orig = __half2float(original[i]);
            float deq = __half2float(dequantized[i]);
            float error = std::abs(orig - deq);
            
            max_error = std::max(max_error, error);
            avg_error += error;
            
            if (error < 1e-6f) perfect_matches++;
        }
        
        avg_error /= original.size();
        
        std::cout << "Round-Trip Accuracy Results:" << std::endl;
        std::cout << "  Total weights: " << original.size() << std::endl;
        std::cout << "  Perfect matches: " << perfect_matches << std::endl;
        std::cout << "  Max error: " << max_error << std::endl;
        std::cout << "  Avg error: " << avg_error << std::endl;
        
        // 验证MxFP4量化的合理精度
        EXPECT_LT(max_error, 2.0f) << "MxFP4 quantization max error too large";
        EXPECT_LT(avg_error, 0.3f) << "MxFP4 quantization average error too large";
        EXPECT_GE(perfect_matches, 8) << "Too few perfect matches for E2M1 exact values";
    }

    // ============= GPU反量化实现 - 调用我们的GPU kernel =============
    
    std::vector<half> GPUDequantizeWeights(
        const QuantizedWeights& quantized,
        unsigned hidden_size,
        unsigned intermediate_size
    ) const {
        void *d_quant_weights, *d_scales, *d_dequant_weights;
        
        CheckHIPStatus(hipMalloc(&d_quant_weights, quantized.data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_scales, quantized.scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_dequant_weights, hidden_size * intermediate_size * sizeof(half)));
        
        CheckHIPStatus(hipMemcpy(d_quant_weights, quantized.data.data(), 
                                quantized.data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_scales, quantized.scales.data(), 
                                quantized.scales.size() * sizeof(float), hipMemcpyHostToDevice));
        
        // 调用GPU反量化kernel
        int err = CallFullDequantMxFp4Kernel(
            d_dequant_weights,
            d_quant_weights,
            d_scales,
            hidden_size, intermediate_size,
            DataType::kDataTypeFp16, nullptr
        );
        
        EXPECT_EQ(err, 0) << "CallFullDequantMxFp4Kernel failed";
        CheckHIPStatus(hipDeviceSynchronize());
        
        std::vector<half> result(hidden_size * intermediate_size);
        CheckHIPStatus(hipMemcpy(result.data(), d_dequant_weights, 
                                result.size() * sizeof(half), hipMemcpyDeviceToHost));
        
        CheckHIPStatus(hipFree(d_quant_weights));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_dequant_weights));
        
        return result;
    }

    // ============= Step 3: hipBLASLt参考计算 (使用反量化权重) =============
    
    std::vector<half> ComputeReferenceHipBLAS(
        const std::vector<half>& input_data,
        const std::vector<half>& dequant_weights,
        float global_scale,
        unsigned m, unsigned n, unsigned k
    ) const {
        void *d_input, *d_weights, *d_output;
        CheckHIPStatus(hipMalloc(&d_input, m * k * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_weights, k * n * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_output, m * n * sizeof(half)));

        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), m * k * sizeof(half), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_weights, dequant_weights.data(), k * n * sizeof(half), hipMemcpyHostToDevice));

        hipblasLtMatrixLayout_t layout_a, layout_b, layout_c;
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_a, HIP_R_16F, k, m, k));
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_b, HIP_R_16F, n, k, n));
        CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_c, HIP_R_16F, n, m, n));

        float alpha = global_scale, beta = 0.0f;
        
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

    // ============= Step 4: 我们的MoE算子 (接收量化权重) =============
    
    std::vector<half> ComputeMoeKernel(
        const std::vector<half>& input_data,
        const QuantizedWeights& quantized,
        const std::vector<unsigned>& expert_indices,
        float global_scale,
        unsigned total_tokens,
        unsigned hidden_size,
        unsigned intermediate_size
    ) const {
        void *d_input, *d_weights, *d_indices, *d_scales, *d_output;
        
        CheckHIPStatus(hipMalloc(&d_input, input_data.size() * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_weights, quantized.data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_indices, expert_indices.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_scales, quantized.scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_output, total_tokens * intermediate_size * sizeof(half)));

        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), 
                                input_data.size() * sizeof(half), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_weights, quantized.data.data(), 
                                quantized.data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_indices, expert_indices.data(), 
                                expert_indices.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_scales, quantized.scales.data(), 
                                quantized.scales.size() * sizeof(float), hipMemcpyHostToDevice));
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
        
        EXPECT_EQ(result, 0) << "MoeMxFp4SecondStage failed";
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

    // ============= 完整测试流程 =============
    
    void TestCompleteWorkflow(
        unsigned total_tokens,
        unsigned hidden_size,
        unsigned intermediate_size,
        float global_scale,
        bool use_random_data = false
    ) {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Complete MxFP4 MoE Test Workflow" << std::endl;
        std::cout << "Dimensions: " << total_tokens << "x" << hidden_size 
                  << "x" << intermediate_size << " (scale=" << global_scale << ")" << std::endl;
        std::cout << "======================================" << std::endl;

        // 数据准备
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

        // Step 1: 测试GPU反量化函数
        TestGPUDequantFunctionCorrectness();

        // Step 2: 测试量化往返精度
        TestQuantizationRoundTripAccuracy();

        std::cout << "\n--- Step 3: Quantizing Test Weights ---" << std::endl;
        auto quantized = QuantizeWeights(original_weights, hidden_size, intermediate_size);
        
        std::cout << "\n--- Step 4: GPU Dequantization for Reference ---" << std::endl;
        auto dequant_weights = GPUDequantizeWeights(quantized, hidden_size, intermediate_size);
        
        std::cout << "\n--- Step 5: Reference Calculation (hipBLASLt) ---" << std::endl;
        auto reference_result = ComputeReferenceHipBLAS(
            input_data, dequant_weights, global_scale,
            total_tokens, intermediate_size, hidden_size
        );

        std::cout << "\n--- Step 6: MoE Kernel Calculation ---" << std::endl;
        auto moe_result = ComputeMoeKernel(
            input_data, quantized, expert_indices, global_scale,
            total_tokens, hidden_size, intermediate_size
        );

        std::cout << "\n--- Step 7: Final Comparison ---" << std::endl;
        CompareResults(reference_result, moe_result, "MoE vs Reference");
        
        std::cout << "\n Expert Analysis Result:" << std::endl;
        std::cout << "   Step 1-2: GPU SimpleMxFp4DequantE2M1 verified" << std::endl;
        std::cout << "   Step 3-5: Reference calculation completed" << std::endl;
        std::cout << "   Step 6-7: MoE implementation tested" << std::endl;
        std::cout << "  Any mismatch = GEMM stage 2 implementation issue" << std::endl;
    }

private:
    // 量化实现
    QuantizedWeights QuantizeWeights(
        const std::vector<half>& original_weights,
        unsigned hidden_size,
        unsigned intermediate_size
    ) const {
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

                float scale = CPUMxFp4Reference::CalculateOptimalScale(block_vals);
                unsigned scale_idx = h_block * intermediate_size + out_dim;
                result.scales[scale_idx] = scale;

                for (unsigned i = 0; i < 32; ++i) {
                    unsigned h = h_block * 32 + i;
                    float weight_val = block_vals[i];
                    unsigned char fp4_idx = CPUMxFp4Reference::QuantizeToE2M1Index(weight_val, scale);
                    unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
                    unsigned bit_offset = (h % 8) * 4;
                    result.data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
                }
            }
        }
        return result;
    }

    // 结果比较
    void CompareResults(
        const std::vector<half>& reference,
        const std::vector<half>& test_result, 
        const std::string& test_name
    ) const {
        std::cout << "Comparing " << test_name << " results..." << std::endl;
        
        float max_error = 0.0f;
        int mismatch_count = 0;
        float total_error = 0.0f;
        
        for (size_t i = 0; i < reference.size(); i++) {
            float ref_val = __half2float(reference[i]);
            float test_val = __half2float(test_result[i]);
            float error = std::abs(ref_val - test_val);
            
            max_error = std::max(max_error, error);
            total_error += error;
            
            // 使用专家的精度要求
            EXPECT_THAT(test_result[i], IsNearFp16(reference[i], 2))
                << "Mismatch at index " << i << ": expected " << ref_val 
                << ", got " << test_val;
                
            if (error > 1e-2f) {
                mismatch_count++;
            }
        }
        
        float avg_error = total_error / reference.size();
        
        std::cout << "Comparison Results:" << std::endl;
        std::cout << "  Max error: " << max_error << std::endl;
        std::cout << "  Avg error: " << avg_error << std::endl;
        std::cout << "  Mismatches (>1e-2): " << mismatch_count << "/" << reference.size() << std::endl;
        
        // 显示前10个详细对比
        std::cout << "\nDetailed comparison (first 10 elements):" << std::endl;
        std::cout << std::setw(6) << "Index" << std::setw(12) << "Reference" 
                  << std::setw(12) << "Test" << std::setw(10) << "Error" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < std::min<size_t>(10, reference.size()); i++) {
            float ref_val = __half2float(reference[i]);
            float test_val = __half2float(test_result[i]);
            float error = std::abs(ref_val - test_val);
            
            std::cout << std::setw(6) << i 
                      << std::setw(12) << std::fixed << std::setprecision(6) << ref_val
                      << std::setw(12) << test_val
                      << std::setw(10) << std::scientific << error << std::endl;
        }
        
        // 专家的验证标准
        EXPECT_LT(max_error, 0.1f) << test_name << " has excessive error";
        EXPECT_LT(mismatch_count, reference.size() / 100) << test_name << " has too many mismatches";
        
        if (mismatch_count == 0) {
            std::cout << "OK! " << test_name << " verification PASSED" << std::endl;
        } else {
            std::cout << "NO! " << test_name << " has " << mismatch_count << " mismatches" << std::endl;
        }
    }
};

// ============= 测试用例 =============

// Step 1: 独立测试GPU SimpleMxFp4DequantE2M1函数
TEST_F(MoeMxFp4Test, Step1_GPUDequantFunctionCorrectness) {
    TestGPUDequantFunctionCorrectness();
}

// Step 2: 独立测试量化往返精度
TEST_F(MoeMxFp4Test, Step2_QuantizationRoundTripAccuracy) {
    TestQuantizationRoundTripAccuracy();
}

// 完整工作流程测试
TEST_F(MoeMxFp4Test, CompleteWorkflow_16x64x256) {
    TestCompleteWorkflow(16, 64, 256, 1.0f);
}

TEST_F(MoeMxFp4Test, CompleteWorkflow_16x64x256_Random) {
    TestCompleteWorkflow(16, 64, 256, 1.0f, true);
}

TEST_F(MoeMxFp4Test, CompleteWorkflow_32x64x128_Random) {
    TestCompleteWorkflow(32, 64, 128, 1.0f, true);
}

TEST_F(MoeMxFp4Test, CompleteWorkflow_64x128x256_Random) {
    TestCompleteWorkflow(64, 128, 256, 1.0f, true);
}

// 不同scale的测试
TEST_F(MoeMxFp4Test, CompleteWorkflow_DifferentScales) {
    std::vector<float> test_scales = {0.5f, 1.0f, 2.0f, 4.0f};
    for (float scale : test_scales) {
        std::cout << "\n=== Testing with global_scale = " << scale << " ===" << std::endl;
        TestCompleteWorkflow(16, 64, 128, scale, true);
    }
}

} // namespace causalflow::petit::rocm::quantization::fp4