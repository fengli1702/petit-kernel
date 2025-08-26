#include "moe_gemm_fp4.h"
#include "mxfp4_types.h"
#include "tests/quantization.h"
#include "utils/hip_helper.h"
#include "hal/device.h"

#include <climits>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// BF16转换辅助函数 - 专家级解决方案
// ============================================================================
namespace {
    // BF16到float转换（通过位操作）
    inline float Bfloat16ToFloat(unsigned short bf16_val) {
        unsigned f32_val = static_cast<unsigned>(bf16_val) << 16;
        return reinterpret_cast<const float&>(f32_val);
    }
    
    // float到BF16转换（通过位操作，简单截断）
    inline unsigned short FloatToBfloat16(float f_val) {
        unsigned f32_bits = reinterpret_cast<const unsigned&>(f_val);
        return static_cast<unsigned short>(f32_bits >> 16);
    }
}

// ============================================================================
// 匹配器定义
// ============================================================================
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
MATCHER_P2(IsNearBf16, ref, mantissa_diff, "") {
    // 使用专家级BF16转换
    float a_f = Bfloat16ToFloat(arg);
    float b_f = Bfloat16ToFloat(ref);
    if (std::abs(a_f - b_f) < std::min<float>(1e-2, fabs(b_f) * 0.01f)) {
        return true;
    }
    int mantissa_a = static_cast<unsigned>(arg) & 0x7f;
    int mantissa_b = static_cast<unsigned>(ref) & 0x7f;
    unsigned other_a = static_cast<unsigned>(arg) & 0x7f80;
    unsigned other_b = static_cast<unsigned>(ref) & 0x7f80;
    bool result = other_a == other_b &&
                  std::abs(mantissa_a - mantissa_b) <= mantissa_diff;
    if (!result && result_listener->IsInterested()) {
        *result_listener << "Expected bfloat16 value near " << std::hex << "0x"
                         << ref << " (" << b_f << "), but got " << std::hex
                         << "0x" << arg << " (" << a_f << ")";
    }
    return result;
}

// ============================================================================
// CPU参考实现
// ============================================================================

class CPUMxFp4Reference {
public:
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
    
    static float CPUDequantE2M1(unsigned fp4_val, float scale) {
        fp4_val = fp4_val & 0xF;
        return cpu_e2m1_lut[fp4_val] * scale;
    }
};

constexpr float CPUMxFp4Reference::cpu_e2m1_lut[16];

// CPU参考激活函数实现
class CPUActivationReference {
public:
    static float GELU(float x) {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float a = 0.044715f;
        float x3 = x * x * x;
        float inner = sqrt_2_over_pi * (x + a * x3);
        return 0.5f * x * (1.0f + tanhf(inner));
    }
    
    static float Swish(float x, float beta = 1.0f) {
        return x / (1.0f + expf(-beta * x));
    }
    
    static float ReLU(float x) {
        return std::max(0.0f, x);
    }
    
    static float ApplyActivation(float x, ActivationType activation) {
        switch (activation) {
            case ActivationType::kGELU: return GELU(x);
            case ActivationType::kSwish: return Swish(x);
            case ActivationType::kReLU: return ReLU(x);
            case ActivationType::kIdentity: return x;
            default: return x;
        }
    }
};

// ============================================================================
// 量化权重结构 - 支持W1和W2
// ============================================================================

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
// 主测试类
// ============================================================================

class MoEMxFp4Test : public ::testing::Test {
public:
    void SetUp() override {
        CheckHIPStatus(hipSetDevice(0));
        CheckHIPStatus(hipMalloc(&d_global_scale_, sizeof(float)));
        
        CheckHipblasStatus(hipblasLtCreate(&hipblaslt_handle_));
        CheckHipblasStatus(hipblasLtMatmulDescCreate(&matmul_desc_, HIPBLAS_COMPUTE_32F, HIP_R_32F));
        CheckHIPStatus(hipMalloc(&d_workspace_, kWorkspaceSize));
        auto plat = hal::GetPlatform("rocm");
        ASSERT_EQ(absl::OkStatus(), plat->GetDevice(0, &dev_));
    }

    void TearDown() override {
        CheckHIPStatus(hipFree(d_global_scale_));
        CheckHIPStatus(hipFree(d_workspace_));
        CheckHipblasStatus(hipblasLtMatmulDescDestroy(matmul_desc_));
        CheckHipblasStatus(hipblasLtDestroy(hipblaslt_handle_));
        CheckHIPStatus(hipDeviceSynchronize());
    }

protected:
    float *d_global_scale_;
    std::unique_ptr<hal::Device> dev_;
    hipblasLtHandle_t hipblaslt_handle_;
    hipblasLtMatmulDesc_t matmul_desc_;
    // hipBLASLt 相关成员变量
    void* d_workspace_;
    static constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;

    static inline void CheckHipblasStatus(hipblasStatus_t status) {
        if (status != HIPBLAS_STATUS_SUCCESS) {
            std::cerr << "HipBLAS Error: " << status << std::endl;
            throw std::runtime_error("HipBLAS Error");
        }
    }
    

    // ============================================================================
    // Step 1: 测试GPU反量化函数
    // ============================================================================
    
    void TestGPUDequantFunctionCorrectness() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 1: Testing GPU SimpleMxFp4DequantE2M1 Function" << std::endl;
        std::cout << "======================================" << std::endl;
        
        // 准备测试数据：所有可能的4-bit值和多种scale
        std::vector<unsigned> fp4_test_values;
        std::vector<float> scale_test_values;
        
        std::vector<float> test_scales = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
        
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
        
        std::cout << "✓ GPU SimpleMxFp4DequantE2M1 Function Test PASSED" << std::endl;
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

    // ============================================================================
    // Step 2: 测试量化→反量化的往返精度  
    // ============================================================================
    
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
        auto gpu_dequantized = GPUDequantizeWeights(
            quantized.w1_data, 
            quantized.w1_scales, 
            hidden_size, 
            intermediate_size
        );
        
        // 分析往返精度
        AnalyzeRoundTripAccuracy(original_weights, gpu_dequantized);
        
        std::cout << "✓ Quantization Round-Trip Accuracy Test PASSED" << std::endl;
    }
    void TestGPUDequantFunctionCorrectnessBF16() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 1: Testing GPU SimpleMxFp4DequantE2M1 Function (BF16)" << std::endl;
        std::cout << "======================================" << std::endl;
        
        std::vector<unsigned> fp4_test_values;
        std::vector<float> scale_test_values;
        std::vector<float> test_scales = {0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
        
        for (float scale : test_scales) {
            for (unsigned fp4 = 0; fp4 < 16; fp4++) {
                fp4_test_values.push_back(fp4);
                scale_test_values.push_back(scale);
            }
        }
        
        unsigned num_tests = fp4_test_values.size();
        
        void *d_fp4_values, *d_scales, *d_gpu_results;
        CheckHIPStatus(hipMalloc(&d_fp4_values, num_tests * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_scales, num_tests * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_gpu_results, num_tests * sizeof(float)));
        
        CheckHIPStatus(hipMemcpy(d_fp4_values, fp4_test_values.data(), num_tests * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_scales, scale_test_values.data(), num_tests * sizeof(float), hipMemcpyHostToDevice));
        
        int err = CallTestGPUDequantKernel(
            reinterpret_cast<float*>(d_gpu_results),
            reinterpret_cast<const unsigned*>(d_fp4_values),
            reinterpret_cast<const float*>(d_scales),
            num_tests, nullptr, DataType::kDataTypeBf16 // Specify BF16
        );
        
        ASSERT_EQ(err, 0) << "CallTestGPUDequantKernel for BF16 failed";
        CheckHIPStatus(hipDeviceSynchronize());
        
        std::vector<float> gpu_results(num_tests);
        CheckHIPStatus(hipMemcpy(gpu_results.data(), d_gpu_results, num_tests * sizeof(float), hipMemcpyDeviceToHost));
        
        std::vector<float> cpu_results(num_tests);
        for (unsigned i = 0; i < num_tests; i++) {
            cpu_results[i] = CPUMxFp4Reference::CPUDequantE2M1(fp4_test_values[i], scale_test_values[i]);
        }
        
        ValidateGPUvsCPUResults(gpu_results, cpu_results, fp4_test_values, scale_test_values);
        
        CheckHIPStatus(hipFree(d_fp4_values)); CheckHIPStatus(hipFree(d_scales)); CheckHIPStatus(hipFree(d_gpu_results));
        
        std::cout << "✓ GPU SimpleMxFp4DequantE2M1 Function Test (BF16) PASSED" << std::endl;
    }
    void AnalyzeRoundTripAccuracyBF16(
        const std::vector<unsigned short>& original,
        const std::vector<unsigned short>& dequantized
    ) {
        float max_error = 0.0f;
        float avg_error = 0.0f;
        int perfect_matches = 0;
        
        for (size_t i = 0; i < original.size(); i++) {
            float orig = Bfloat16ToFloat(original[i]);
            float deq = Bfloat16ToFloat(dequantized[i]);
            float error = std::abs(orig - deq);
            
            max_error = std::max(max_error, error);
            avg_error += error;
            
            if (error < 1e-4f) perfect_matches++; // Looser tolerance for perfect match
        }
        
        avg_error /= original.size();
        
        std::cout << "Round-Trip Accuracy Results (BF16):" << std::endl;
        std::cout << "  Total weights: " << original.size() << std::endl;
        std::cout << "  Perfect matches: " << perfect_matches << std::endl;
        std::cout << "  Max error: " << max_error << std::endl;
        std::cout << "  Avg error: " << avg_error << std::endl;
        
        // BF16 has lower precision, so we expect larger errors
        EXPECT_LT(max_error, 2.5f);
        EXPECT_LT(avg_error, 0.5f);
        EXPECT_GE(perfect_matches, 8);
    }

    void TestQuantizationRoundTripAccuracyBF16() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 2: Testing Quantization Round-Trip Accuracy (BF16)" << std::endl;
        std::cout << "======================================" << std::endl;
        
        unsigned hidden_size = 64, intermediate_size = 32;
        std::vector<unsigned short> original_weights(hidden_size * intermediate_size);
        
        std::vector<float> exact_values = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f };
        
        std::mt19937 gen(42);
        std::normal_distribution<float> random_dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < original_weights.size(); i++) {
            float val = (i < exact_values.size()) ? exact_values[i] : random_dist(gen);
            original_weights[i] = FloatToBfloat16(val);
        }
        
        auto quantized = QuantizeWeightsBF16(original_weights, hidden_size, intermediate_size);
        auto gpu_dequantized = GPUDequantizeWeightsBF16(quantized.w1_data, quantized.w1_scales, hidden_size, intermediate_size);
        AnalyzeRoundTripAccuracyBF16(original_weights, gpu_dequantized);
        
        std::cout << "✓ Quantization Round-Trip Accuracy Test (BF16) PASSED" << std::endl;
    }
    QuantizedFFNWeights QuantizeWeights(
        const std::vector<half>& original_weights,
        unsigned input_dim, unsigned output_dim
    ) {
        QuantizedFFNWeights result;
        
        result.w1_data.resize((input_dim * output_dim + 7) / 8, 0);
        result.w1_scales.resize((input_dim / kMxFp4BlockSize) * output_dim);
        
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
                result.w1_scales[scale_idx] = scale;

                for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                    unsigned in_idx = in_block * kMxFp4BlockSize + i;
                    float weight_val = block_vals[i];
                    unsigned char fp4_idx = CPUMxFp4Reference::QuantizeToE2M1Index(weight_val, scale);
                    unsigned packed_idx = out_dim * packed_cols_stride + (in_idx / 8);
                    unsigned bit_offset = (in_idx % 8) * 4;
                    result.w1_data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
                }
            }
        }
        
        return result;
    }
    QuantizedFFNWeights QuantizeWeightsBF16(
        const std::vector<unsigned short>& original_weights,
        unsigned input_dim, unsigned output_dim
    ) {
        QuantizedFFNWeights result;
        result.w1_data.resize((input_dim * output_dim + 7) / 8, 0);
        result.w1_scales.resize((input_dim / kMxFp4BlockSize) * output_dim);
        
        for (unsigned out_dim = 0; out_dim < output_dim; ++out_dim) {
            for (unsigned in_block = 0; in_block < input_dim / kMxFp4BlockSize; ++in_block) {
                std::vector<float> block_vals(kMxFp4BlockSize);
                for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                    unsigned in_idx = in_block * kMxFp4BlockSize + i;
                    unsigned weight_idx = in_idx * output_dim + out_dim;
                    block_vals[i] = Bfloat16ToFloat(original_weights[weight_idx]);
                }

                float scale = CPUMxFp4Reference::CalculateOptimalScale(block_vals);
                result.w1_scales[in_block * output_dim + out_dim] = scale;

                for (unsigned i = 0; i < kMxFp4BlockSize; ++i) {
                    unsigned in_idx = in_block * kMxFp4BlockSize + i;
                    unsigned char fp4_idx = CPUMxFp4Reference::QuantizeToE2M1Index(block_vals[i], scale);
                    unsigned packed_idx = out_dim * (input_dim / 8) + (in_idx / 8);
                    unsigned bit_offset = (in_idx % 8) * 4;
                    result.w1_data[packed_idx] |= (static_cast<unsigned>(fp4_idx & 0xF) << bit_offset);
                }
            }
        }
        return result;
    }
    std::vector<half> GPUDequantizeWeights(
        const std::vector<unsigned>& quantized_data, // 直接传入数据
        const std::vector<float>& scales,           // 直接传入 scales
        unsigned input_dim,
        unsigned output_dim
    ) const {
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
            input_dim, // 注意维度顺序
            output_dim,
            DataType::kDataTypeFp16, nullptr
        );

        EXPECT_EQ(err, 0) << "CallFullDequantMxFp4Kernel failed";
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
        const std::vector<unsigned>& quantized_data, // 直接传入数据
        const std::vector<float>& scales,           // 直接传入 scales
        unsigned input_dim,
        unsigned output_dim
    ) const {
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
            input_dim, // 注意维度顺序
            output_dim,
            DataType::kDataTypeBf16, nullptr
        );

        EXPECT_EQ(err, 0) << "CallFullDequantMxFp4Kernel for BF16 failed";
        CheckHIPStatus(hipDeviceSynchronize());

        std::vector<unsigned short> result(input_dim * output_dim);
        CheckHIPStatus(hipMemcpy(result.data(), d_dequant_weights, 
                                result.size() * sizeof(unsigned short), hipMemcpyDeviceToHost));

        CheckHIPStatus(hipFree(d_quant_weights));
        CheckHIPStatus(hipFree(d_scales));
        CheckHIPStatus(hipFree(d_dequant_weights));

        return result;
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

    // ============================================================================
    // Step 3: 测试激活函数
    // ============================================================================
    
    void TestActivationFunctionCorrectness() {
        std::cout << "\n======================================" << std::endl;
        std::cout << "Step 3: Testing Activation Functions" << std::endl;
        std::cout << "======================================" << std::endl;
        
        const unsigned num_tests = 1000;
        std::vector<half> test_input(num_tests);
        
        // 生成测试数据：包含负值、零值、正值
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 2.0f);
        for (size_t i = 0; i < num_tests; i++) {
            test_input[i] = __float2half(dist(gen));
        }
        
        // 测试三种激活函数
        std::vector<ActivationType> activations = {
            ActivationType::kGELU, 
            ActivationType::kSwish, 
            ActivationType::kReLU
        };
        
        for (auto activation : activations) {
            std::cout << "Testing activation: " << static_cast<int>(activation) << std::endl;
            TestSingleActivationFunction(test_input, activation);
        }
        
        std::cout << "✓ Activation Functions Test PASSED" << std::endl;
    }

    void TestSingleActivationFunction(
        const std::vector<half>& test_input, 
        ActivationType activation
    ) {
        const unsigned num_tests = test_input.size();
        
        // 分配GPU内存
        void *d_input, *d_output;
        CheckHIPStatus(hipMalloc(&d_input, num_tests * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_output, num_tests * sizeof(half)));
        
        CheckHIPStatus(hipMemcpy(d_input, test_input.data(), 
                                num_tests * sizeof(half), hipMemcpyHostToDevice));
        
        // 调用GPU激活函数kernel
        int err = CallTestActivationKernel(
            d_output, d_input, num_tests, 
            ActivationTypeToInt(activation), DataType::kDataTypeFp16, nullptr
        );
        
        ASSERT_EQ(err, 0) << "CallTestActivationKernel failed";
        CheckHIPStatus(hipDeviceSynchronize());
        
        // 获取GPU结果
        std::vector<half> gpu_results(num_tests);
        CheckHIPStatus(hipMemcpy(gpu_results.data(), d_output, 
                                num_tests * sizeof(half), hipMemcpyDeviceToHost));
        
        // CPU参考计算
        std::vector<float> cpu_results(num_tests);
        for (size_t i = 0; i < num_tests; i++) {
            float input_val = __half2float(test_input[i]);
            cpu_results[i] = CPUActivationReference::ApplyActivation(input_val, activation);
        }
        
        // 验证结果
        ValidateActivationResults(gpu_results, cpu_results, activation);
        
        CheckHIPStatus(hipFree(d_input));
        CheckHIPStatus(hipFree(d_output));
    }

    void ValidateActivationResults(
        const std::vector<half>& gpu_results,
        const std::vector<float>& cpu_results,
        ActivationType activation
    ) {
        int   mismatches = 0;
        float max_diff = 0.0f;
        float total_diff = 0.0f;
        
        std::cout << "\nActivation " << static_cast<int>(activation) << " Results:" << std::endl;
        std::cout << std::setw(6) << "Index" << std::setw(12) << "CPU" 
                  << std::setw(12) << "GPU" << std::setw(10) << "Diff" << std::endl;
        std::cout << std::string(50, '-') << std::endl;
        
        for (size_t i = 0; i < std::min<size_t>(10, gpu_results.size()); i++) {
            float gpu_val = __half2float(gpu_results[i]);
            float cpu_val = cpu_results[i];
            float diff = std::abs(gpu_val - cpu_val);
            
            std::cout << std::setw(6) << i 
                      << std::setw(12) << std::fixed << std::setprecision(6) << cpu_val
                      << std::setw(12) << gpu_val
                      << std::setw(10) << std::scientific << diff << std::endl;
        }
        
        for (size_t i = 0; i < gpu_results.size(); i++) {
            float gpu_val = __half2float(gpu_results[i]);
            float cpu_val = cpu_results[i];
            float diff = std::abs(gpu_val - cpu_val);
            
            max_diff = std::max(max_diff, diff);
            total_diff += diff;
            
            // 激活函数的容差根据函数类型设定
            float tolerance = 1e-3f;
            if (activation == ActivationType::kGELU) {
                tolerance = 5e-3f; // GELU近似，允许更大误差
            }
            if (activation == ActivationType::kSwish) {
                tolerance = 5e-3f; // kSwish近似，允许更大误差
            }
            
            if (diff > tolerance) {
                mismatches++;
            }
        }
        
        float avg_diff = total_diff / gpu_results.size();
        
        std::cout << "\nActivation " << static_cast<int>(activation) << " Summary:" << std::endl;
        std::cout << "  Max difference: " << max_diff << std::endl;
        std::cout << "  Avg difference: " << avg_diff << std::endl;
        std::cout << "  Mismatches: " << mismatches << "/" << gpu_results.size() << std::endl;
        
        // 验证标准
        EXPECT_LT(mismatches, static_cast<int>(gpu_results.size() * 0.01f)) 
            << "Too many activation mismatches";
        EXPECT_LT(max_diff, 0.01f) << "Activation max error too large";
    }

    // ============================================================================
    // Step 4: 完整FFN流程测试 - 支持FP16和BF16
    // ============================================================================
    
    void TestCompleteFFNPipeline(const MoEStage2Config& config, float global_scale, bool use_random_data = true) {
        if (config.input_type == DataType::kDataTypeBf16) {
            TestCompleteFFNPipelineBF16(config, global_scale, use_random_data);
        } else {
            TestCompleteFFNPipelineFP16(config, global_scale, use_random_data);
        }
    }

    void TestCompleteFFNPipelineFP16(const MoEStage2Config& config, float global_scale, bool use_random_data) {
        std::cout << "\n======================================" << std::endl;
        //std::cout << "Step 4: Complete FFN Pipeline Test (FP16)" << std::endl;
        std::cout << "Config: " << config.total_tokens << "x" << config.hidden_size 
                  << "x" << config.intermediate_size << ", activation=" << static_cast<int>(config.activation) << std::endl;
        std::cout << "======================================" << std::endl;

        //// Step 1-3: 基础函数测试
        //TestGPUDequantFunctionCorrectness();
        //TestQuantizationRoundTripAccuracy();
        //TestActivationFunctionCorrectness();

        // Step 4: 准备测试数据
        auto [input_data, ffn_weights, expert_indices] = PrepareFFNTestDataFP16(config, use_random_data);

        // Step 5: 计算CPU参考结果
        //std::cout << "\n--- Computing CPU Reference ---" << std::endl;
        auto reference_result = ComputeHipBLASLtCompleteFFNFP16(
            input_data, ffn_weights, expert_indices, config, global_scale
        );

        // Step 6: 计算GPU结果
        //std::cout << "\n--- Computing GPU Complete FFN ---" << std::endl;
        auto gpu_result = ComputeGPUCompleteFFNFP16(
            input_data, ffn_weights, expert_indices, config, global_scale
        );

        // Step 7: 比较结果
        std::cout << "\n--- Comparing Complete FFN Results ---" << std::endl;
        CompareResultsFP16(reference_result, gpu_result, "Complete FFN");
        
        std::cout << "\n✓ Complete FFN Pipeline Test PASSED" << std::endl;
    }

    void TestCompleteFFNPipelineBF16(const MoEStage2Config& config, float global_scale, bool use_random_data) {
        std::cout << "\n======================================" << std::endl;
        //std::cout << "Step 4: Complete FFN Pipeline Test (BF16)" << std::endl;
        std::cout << "Config: " << config.total_tokens << "x" << config.hidden_size 
                  << "x" << config.intermediate_size << ", activation=" << static_cast<int>(config.activation) << std::endl;
        std::cout << "======================================" << std::endl;

        // BF16数据准备 - 使用unsigned short表示BF16
        std::vector<unsigned short> input_data(config.total_tokens * config.hidden_size);
        std::vector<unsigned short> w1_original(config.hidden_size * config.intermediate_size);
        std::vector<unsigned short> w2_original(config.intermediate_size * config.hidden_size);
        std::vector<unsigned> expert_indices(config.total_tokens, 0); // 单专家测试

        std::mt19937 gen(42);
        if (use_random_data) {
            std::normal_distribution<float> input_dist(0.0f, 0.3f);
            std::normal_distribution<float> weight_dist(0.0f, 0.2f);

            for (auto& val : input_data) val = FloatToBfloat16(input_dist(gen));
            for (auto& val : w1_original) val = FloatToBfloat16(weight_dist(gen));
            for (auto& val : w2_original) val = FloatToBfloat16(weight_dist(gen));
        } else {
            // 使用与FP16非随机测试类似的大数值
            for (size_t i = 0; i < input_data.size(); i++) {
                input_data[i] = FloatToBfloat16(1.0f + 0.5f * (i % 10));
            }
            for (size_t i = 0; i < w1_original.size(); i++) {
                w1_original[i] = FloatToBfloat16(0.5f + 0.2f * (i % 20));
            }
            for (size_t i = 0; i < w2_original.size(); i++) {
                w2_original[i] = FloatToBfloat16(0.3f + 0.1f * (i % 15));
            }
        }

        // 量化权重
        QuantizedFFNWeights ffn_weights = QuantizeFFNWeightsBF16(w1_original, w2_original, config);

        // CPU参考计算
        auto reference_result = ComputeHipBLASLtCompleteFFNBF16(
            input_data, ffn_weights, expert_indices, config, global_scale
        );

        // GPU计算
        auto gpu_result = ComputeGPUCompleteFFNBF16(
            input_data, ffn_weights, expert_indices, config, global_scale
        );

        // 比较结果
        CompareResultsBF16(reference_result, gpu_result, "Complete FFN BF16");

        std::cout << "\n✓ Complete FFN Pipeline Test (BF16) PASSED" << std::endl;
    }

    // ============================================================================
    // 测试数据准备 - FP16版本
    // ============================================================================
    
    std::tuple<std::vector<half>, QuantizedFFNWeights, std::vector<unsigned>> 
    PrepareFFNTestDataFP16(const MoEStage2Config& config, bool use_random_data) {
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
        QuantizedFFNWeights ffn_weights = QuantizeFFNWeightsFP16(w1_original, w2_original, config);
        
        return std::make_tuple(input_data, ffn_weights, expert_indices);
    }

    QuantizedFFNWeights QuantizeFFNWeightsFP16(
        const std::vector<half>& w1_weights,
        const std::vector<half>& w2_weights, 
        const MoEStage2Config& config
    ) {
        QuantizedFFNWeights result;
        
        // 量化W1 (hidden_size -> intermediate_size)
        result.w1_data.resize((config.hidden_size * config.intermediate_size + 7) / 8, 0);
        result.w1_scales.resize((config.hidden_size / kMxFp4BlockSize) * config.intermediate_size);
        QuantizeWeightsLayerFP16(w1_weights, result.w1_data, result.w1_scales, 
                           config.hidden_size, config.intermediate_size);
        
        // 量化W2 (intermediate_size -> hidden_size)
        result.w2_data.resize((config.intermediate_size * config.hidden_size + 7) / 8, 0);
        result.w2_scales.resize((config.intermediate_size / kMxFp4BlockSize) * config.hidden_size);
        QuantizeWeightsLayerFP16(w2_weights, result.w2_data, result.w2_scales,
                           config.intermediate_size, config.hidden_size);
        
        // 设置元数据
        result.w1_metadata = MxFp4WeightMetadata::CalculateMetadata(
            config.hidden_size, config.intermediate_size, config.input_type);
        result.w2_metadata = MxFp4WeightMetadata::CalculateMetadata(
            config.intermediate_size, config.hidden_size, config.input_type);
        
        return result;
    }

    QuantizedFFNWeights QuantizeFFNWeightsBF16(
        const std::vector<unsigned short>& w1_weights,
        const std::vector<unsigned short>& w2_weights, 
        const MoEStage2Config& config
    ) {
        QuantizedFFNWeights result;
        
        // 量化W1 (hidden_size -> intermediate_size)
        result.w1_data.resize((config.hidden_size * config.intermediate_size + 7) / 8, 0);
        result.w1_scales.resize((config.hidden_size / kMxFp4BlockSize) * config.intermediate_size);
        QuantizeWeightsLayerBF16(w1_weights, result.w1_data, result.w1_scales, 
                           config.hidden_size, config.intermediate_size);
        
        // 量化W2 (intermediate_size -> hidden_size)
        result.w2_data.resize((config.intermediate_size * config.hidden_size + 7) / 8, 0);
        result.w2_scales.resize((config.intermediate_size / kMxFp4BlockSize) * config.hidden_size);
        QuantizeWeightsLayerBF16(w2_weights, result.w2_data, result.w2_scales,
                           config.intermediate_size, config.hidden_size);
        
        // 设置元数据
        result.w1_metadata = MxFp4WeightMetadata::CalculateMetadata(
            config.hidden_size, config.intermediate_size, config.input_type);
        result.w2_metadata = MxFp4WeightMetadata::CalculateMetadata(
            config.intermediate_size, config.hidden_size, config.input_type);
        
        return result;
    }

    void QuantizeWeightsLayerFP16(
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

    void QuantizeWeightsLayerBF16(
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
                    block_vals[i] = Bfloat16ToFloat(original_weights[weight_idx]);
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

    // ============================================================================
    // 使用 hipBLASLt 的完整 FFN 实现
    // ============================================================================
    
    std::vector<half> ComputeHipBLASLtCompleteFFNFP16(
        const std::vector<half>& input_data,
        const QuantizedFFNWeights& ffn_weights,
        const std::vector<unsigned>& expert_indices, // (未被使用，但保持签名一致)
        const MoEStage2Config& config,
        float global_scale
    ) {
        std::cout << "Computing hipBLASLt Complete FFN (FP16)..." << std::endl;

        // Step 1: 在函数内部反量化权重
        auto w1_dequant = CPUDequantizeWeightsFP16(ffn_weights.w1_data, ffn_weights.w1_scales, config.hidden_size, config.intermediate_size);
        auto w2_dequant = CPUDequantizeWeightsFP16(ffn_weights.w2_data, ffn_weights.w2_scales, config.intermediate_size, config.hidden_size);
    
        // 维度定义
        const unsigned m = config.total_tokens;
        const unsigned k1 = config.hidden_size;
        const unsigned n1 = config.intermediate_size;
        const unsigned k2 = config.intermediate_size;
        const unsigned n2 = config.hidden_size;
        size_t type_size = sizeof(half);

        // GPU 内存分配
        void *d_input, *d_w1, *d_w2, *d_inter1, *d_inter2, *d_output;
        CheckHIPStatus(hipMalloc(&d_input, m * k1 * type_size));
        CheckHIPStatus(hipMalloc(&d_w1, k1 * n1 * type_size));
        CheckHIPStatus(hipMalloc(&d_w2, k2 * n2 * type_size));
        CheckHIPStatus(hipMalloc(&d_inter1, m * n1 * type_size));
        CheckHIPStatus(hipMalloc(&d_inter2, m * k2 * type_size));
        CheckHIPStatus(hipMalloc(&d_output, m * n2 * type_size));

        // 数据拷贝
        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), m * k1 * type_size, hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1, w1_dequant.data(), k1 * n1 * type_size, hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2, w2_dequant.data(), k2 * n2 * type_size, hipMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;

        // 重置转置标志，确保每次调用都是干净的状态
        hipblasOperation_t no_trans = HIPBLAS_OP_N;
        CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)));
        CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)));

        // ===== Step 2: hipBLASLt GEMM - X @ W1 = Intermediate =====
        // 使用 C^T = B^T * A^T 技巧处理行主序矩阵
        {
            hipblasLtMatrixLayout_t layout_A, layout_B, layout_C;
            // B (W1): [k1, n1], ld=n1
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16F, n1, k1, n1));
            // A (input): [m, k1], ld=k1
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16F, k1, m, k1));
            // C (intermediate): [m, n1], ld=n1
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16F, n1, m, n1));

            CheckHipblasStatus(hipblasLtMatmul(hipblaslt_handle_, matmul_desc_, &alpha, d_w1, layout_B, d_input, layout_A, &beta, d_inter1, layout_C, d_inter1, layout_C, nullptr, d_workspace_, kWorkspaceSize, nullptr));

            hipblasLtMatrixLayoutDestroy(layout_A);
            hipblasLtMatrixLayoutDestroy(layout_B);
            hipblasLtMatrixLayoutDestroy(layout_C);
        }

        // ===== Step 3: 激活函数 =====
        CallTestActivationKernel(d_inter2, d_inter1, m * n1, ActivationTypeToInt(config.activation), config.input_type, nullptr);
        CheckHIPStatus(hipDeviceSynchronize());

        // ===== Step 4: hipBLASLt GEMM - Intermediate @ W2 = Output =====
        alpha = global_scale;
        {
            hipblasLtMatrixLayout_t layout_A, layout_B, layout_C;
            // B (W2): [k2, n2], ld=n2
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16F, n2, k2, n2));
            // A (H_act): [m, k2], ld=k2
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16F, k2, m, k2));
            // C (output): [m, n2], ld=n2
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16F, n2, m, n2));

            CheckHipblasStatus(hipblasLtMatmul(hipblaslt_handle_, matmul_desc_, &alpha, d_w2, layout_B, d_inter2, layout_A, &beta, d_output, layout_C, d_output, layout_C, nullptr, d_workspace_, kWorkspaceSize, nullptr));

            hipblasLtMatrixLayoutDestroy(layout_A);
            hipblasLtMatrixLayoutDestroy(layout_B);
            hipblasLtMatrixLayoutDestroy(layout_C);
        }

        // 获取结果和清理
        std::vector<half> result(m * n2);
        CheckHIPStatus(hipMemcpy(result.data(), d_output, m * n2 * type_size, hipMemcpyDeviceToHost));
        CheckHIPStatus(hipFree(d_input)); CheckHIPStatus(hipFree(d_w1)); CheckHIPStatus(hipFree(d_w2));
        CheckHIPStatus(hipFree(d_inter1)); CheckHIPStatus(hipFree(d_inter2)); CheckHIPStatus(hipFree(d_output));
        return result;
    }

// ============================================================================
// BF16 版本的 hipBLASLt FFN 实现
// ============================================================================
    std::vector<unsigned short> ComputeHipBLASLtCompleteFFNBF16(
        const std::vector<unsigned short>& input_data,
        const QuantizedFFNWeights& ffn_weights,
        const std::vector<unsigned>& expert_indices, // (未被使用，但保持签名一致)
        const MoEStage2Config& config,
        float global_scale
    ) {
        std::cout << "Computing hipBLASLt Complete FFN (BF16)..." << std::endl;

        // Step 1: 在函数内部反量化权重
        auto w1_dequant = CPUDequantizeWeightsBF16(ffn_weights.w1_data, ffn_weights.w1_scales, config.hidden_size, config.intermediate_size);
        auto w2_dequant = CPUDequantizeWeightsBF16(ffn_weights.w2_data, ffn_weights.w2_scales, config.intermediate_size, config.hidden_size);
    
        // 维度定义
        const unsigned m = config.total_tokens;
        const unsigned k1 = config.hidden_size;
        const unsigned n1 = config.intermediate_size;
        const unsigned k2 = config.intermediate_size;
        const unsigned n2 = config.hidden_size;
        size_t type_size = sizeof(unsigned short);

        // GPU 内存分配
        void *d_input, *d_w1, *d_w2, *d_inter1, *d_inter2, *d_output;
        CheckHIPStatus(hipMalloc(&d_input, m * k1 * type_size));
        CheckHIPStatus(hipMalloc(&d_w1, k1 * n1 * type_size));
        CheckHIPStatus(hipMalloc(&d_w2, k2 * n2 * type_size));
        CheckHIPStatus(hipMalloc(&d_inter1, m * n1 * type_size));
        CheckHIPStatus(hipMalloc(&d_inter2, m * k2 * type_size));
        CheckHIPStatus(hipMalloc(&d_output, m * n2 * type_size));

        // 数据拷贝
        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), m * k1 * type_size, hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1, w1_dequant.data(), k1 * n1 * type_size, hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2, w2_dequant.data(), k2 * n2 * type_size, hipMemcpyHostToDevice));

        float alpha = 1.0f, beta = 0.0f;

        // 重置转置标志
        hipblasOperation_t no_trans = HIPBLAS_OP_N;
        CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSA, &no_trans, sizeof(no_trans)));
        CheckHipblasStatus(hipblasLtMatmulDescSetAttribute(matmul_desc_, HIPBLASLT_MATMUL_DESC_TRANSB, &no_trans, sizeof(no_trans)));

        // ===== Step 2: BF16 GEMM - X @ W1 =====
        {
            hipblasLtMatrixLayout_t layout_A, layout_B, layout_C;
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16BF, n1, k1, n1));
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16BF, k1, m, k1));
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16BF, n1, m, n1));
            CheckHipblasStatus(hipblasLtMatmul(hipblaslt_handle_, matmul_desc_, &alpha, d_w1, layout_B, d_input, layout_A, &beta, d_inter1, layout_C, d_inter1, layout_C, nullptr, d_workspace_, kWorkspaceSize, nullptr));
            hipblasLtMatrixLayoutDestroy(layout_A); hipblasLtMatrixLayoutDestroy(layout_B); hipblasLtMatrixLayoutDestroy(layout_C);
        }

        // ===== Step 3: 激活函数 =====
        CallTestActivationKernel(d_inter2, d_inter1, m * n1, ActivationTypeToInt(config.activation), config.input_type, nullptr);
        CheckHIPStatus(hipDeviceSynchronize());

        // ===== Step 4: BF16 GEMM - Intermediate @ W2 =====
        alpha = global_scale;
        {
            hipblasLtMatrixLayout_t layout_A, layout_B, layout_C;
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_16BF, n2, k2, n2));
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_16BF, k2, m, k2));
            CheckHipblasStatus(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_16BF, n2, m, n2));
            CheckHipblasStatus(hipblasLtMatmul(hipblaslt_handle_, matmul_desc_, &alpha, d_w2, layout_B, d_inter2, layout_A, &beta, d_output, layout_C, d_output, layout_C, nullptr, d_workspace_, kWorkspaceSize, nullptr));
            hipblasLtMatrixLayoutDestroy(layout_A); hipblasLtMatrixLayoutDestroy(layout_B); hipblasLtMatrixLayoutDestroy(layout_C);
        }

        // 获取结果和清理
        std::vector<unsigned short> result(m * n2);
        CheckHIPStatus(hipMemcpy(result.data(), d_output, m * n2 * type_size, hipMemcpyDeviceToHost));
        CheckHIPStatus(hipFree(d_input)); CheckHIPStatus(hipFree(d_w1)); CheckHIPStatus(hipFree(d_w2));
        CheckHIPStatus(hipFree(d_inter1)); CheckHIPStatus(hipFree(d_inter2)); CheckHIPStatus(hipFree(d_output));
        return result;
    }

    std::vector<half> CPUDequantizeWeightsFP16(
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

    std::vector<unsigned short> CPUDequantizeWeightsBF16(
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
                dequant_weights[in_dim * output_dim + out_dim] = FloatToBfloat16(weight_val);
            }
        }
        
        return dequant_weights;
    }

    // ============================================================================
    // GPU完整FFN实现 - FP16和BF16版本
    // ============================================================================
    
    std::vector<half> ComputeGPUCompleteFFNFP16(
        const std::vector<half>& input_data,
        const QuantizedFFNWeights& ffn_weights,
        const std::vector<unsigned>& expert_indices,
        const MoEStage2Config& config,
        float global_scale
    ) {
        void *d_input, *d_w1_weights, *d_w2_weights, *d_indices, 
             *d_w1_scales, *d_w2_scales, *d_output;
        
        // 分配GPU内存
        CheckHIPStatus(hipMalloc(&d_input, input_data.size() * sizeof(half)));
        CheckHIPStatus(hipMalloc(&d_w1_weights, ffn_weights.w1_data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_w2_weights, ffn_weights.w2_data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_indices, expert_indices.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_w1_scales, ffn_weights.w1_scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_w2_scales, ffn_weights.w2_scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_output, config.total_tokens * config.hidden_size * sizeof(unsigned short)));
        
        // 复制数据到GPU
        CheckHIPStatus(hipMemcpy(d_input, input_data.data(), 
                                input_data.size() * sizeof(half), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1_weights, ffn_weights.w1_data.data(), 
                                ffn_weights.w1_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2_weights, ffn_weights.w2_data.data(), 
                                ffn_weights.w2_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_indices, expert_indices.data(), 
                                expert_indices.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1_scales, ffn_weights.w1_scales.data(), 
                                ffn_weights.w1_scales.size() * sizeof(float), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2_scales, ffn_weights.w2_scales.data(), 
                                ffn_weights.w2_scales.size() * sizeof(float), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_global_scale_, &global_scale, sizeof(float), hipMemcpyHostToDevice));
        
        // 调用完整FFN kernel
        int result = MoECompleteFFNStage2(
            d_output,
            d_input,
            d_w1_weights,
            d_w2_weights,
            reinterpret_cast<const unsigned*>(d_indices),
            d_w1_scales,
            d_w2_scales,
            d_global_scale_,
            config,
            nullptr
        );
        
        EXPECT_EQ(result, 0) << "MoECompleteFFNStage2 failed";
        CheckHIPStatus(hipDeviceSynchronize());
        
        // 获取结果
        std::vector<half> gpu_result(config.total_tokens * config.hidden_size);
        CheckHIPStatus(hipMemcpy(gpu_result.data(), d_output, 
                                gpu_result.size() * sizeof(half), hipMemcpyDeviceToHost));
        
        // 清理内存
        CheckHIPStatus(hipFree(d_input));
        CheckHIPStatus(hipFree(d_w1_weights));
        CheckHIPStatus(hipFree(d_w2_weights));
        CheckHIPStatus(hipFree(d_indices));
        CheckHIPStatus(hipFree(d_w1_scales));
        CheckHIPStatus(hipFree(d_w2_scales));
        CheckHIPStatus(hipFree(d_output));
        
        return gpu_result;
    }
    std::vector<unsigned short> ComputeGPUCompleteFFNBF16(
        const std::vector<unsigned short>& input_data,
        const QuantizedFFNWeights& ffn_weights,
        const std::vector<unsigned>& expert_indices,
        const MoEStage2Config& config,
        float global_scale
    ) {
        void *d_input, *d_w1_weights, *d_w2_weights, *d_indices,
             *d_w1_scales, *d_w2_scales, *d_output;

        // 分配GPU内存
        CheckHIPStatus(hipMalloc(&d_input, input_data.size() * sizeof(unsigned short)));
        CheckHIPStatus(hipMalloc(&d_w1_weights, ffn_weights.w1_data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_w2_weights, ffn_weights.w2_data.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_indices, expert_indices.size() * sizeof(unsigned)));
        CheckHIPStatus(hipMalloc(&d_w1_scales, ffn_weights.w1_scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_w2_scales, ffn_weights.w2_scales.size() * sizeof(float)));
        CheckHIPStatus(hipMalloc(&d_output, config.total_tokens * config.hidden_size * sizeof(unsigned short)));

        // 复制数据到GPU
        CheckHIPStatus(hipMemcpy(d_input, input_data.data(),
                                input_data.size() * sizeof(unsigned short), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1_weights, ffn_weights.w1_data.data(),
                                ffn_weights.w1_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2_weights, ffn_weights.w2_data.data(),
                                ffn_weights.w2_data.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_indices, expert_indices.data(),
                                expert_indices.size() * sizeof(unsigned), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w1_scales, ffn_weights.w1_scales.data(),
                                ffn_weights.w1_scales.size() * sizeof(float), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_w2_scales, ffn_weights.w2_scales.data(),
                                ffn_weights.w2_scales.size() * sizeof(float), hipMemcpyHostToDevice));
        CheckHIPStatus(hipMemcpy(d_global_scale_, &global_scale, sizeof(float), hipMemcpyHostToDevice));

        // 调用完整FFN kernel
        int result = MoECompleteFFNStage2(d_output, d_input, d_w1_weights, d_w2_weights,
            reinterpret_cast<const unsigned*>(d_indices), d_w1_scales, d_w2_scales,
            d_global_scale_, config, nullptr);
        EXPECT_EQ(result, 0) << "MoECompleteFFNStage2 failed for BF16";
        CheckHIPStatus(hipDeviceSynchronize());

        // 获取结果
        std::vector<unsigned short> gpu_result(config.total_tokens * config.hidden_size);
        CheckHIPStatus(hipMemcpy(gpu_result.data(), d_output, gpu_result.size() * sizeof(unsigned short), hipMemcpyDeviceToHost));

        // 清理内存... (省略以保持简洁)
        return gpu_result;
    }

    // ============================================================================
    // 结果比较和验证 - FP16和BF16版本
    // ============================================================================
    
    void CompareResultsFP16(
        const std::vector<half>& reference,
        const std::vector<half>& test_result,
        const std::string& test_name
        ) {
        std::cout << "Comparing " << test_name << " results (FP16)..." << std::endl;

        ASSERT_EQ(reference.size(), test_result.size());
        if (reference.empty()) {
            std::cout << "Comparison skipped for empty vectors." << std::endl;
            return;
        }
        //打印两个结果（高精度显示）
        //std::cout << std::fixed << std::setprecision(8); // 设置为8位小数精度
        //for (size_t i = 0; i < 10; i++) {
        //    float ref_val = __half2float(reference[i]);
        //    float test_val = __half2float(test_result[i]);
        //    float diff = std::abs(ref_val - test_val);
        //    std::cout << "Index " << std::setw(2) << i 
        //              << ": Reference = " << std::setw(12) << ref_val 
        //              << ", Test = " << std::setw(12) << test_val
        //              << ", Diff = " << std::scientific << diff << std::fixed << std::endl;
        //}
        //std::cout << std::defaultfloat; // 恢复默认格式

        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f; // Will only be calculated for non-tiny reference values
        float total_abs_error = 0.0f;
        float ref_norm = 0.0f, test_norm = 0.0f, dot_product = 0.0f;

        for (size_t i = 0; i < reference.size(); i++) {
            float ref_val = __half2float(reference[i]);
            float test_val = __half2float(test_result[i]);
            float abs_error = std::abs(ref_val - test_val);
            total_abs_error += abs_error;
            max_abs_error = std::max(max_abs_error, abs_error);

            if (std::abs(ref_val) > 1e-4f) { // Guard against division by near-zero
                max_rel_error = std::max(max_rel_error, abs_error / std::abs(ref_val));
            }

            dot_product += ref_val * test_val;
            ref_norm += ref_val * ref_val;
            test_norm += test_val * test_val;
        }

        float avg_abs_error = total_abs_error / reference.size();
        float cosine_sim = 1.0f;
        if (ref_norm > 1e-9f && test_norm > 1e-9f) {
            cosine_sim = dot_product / (sqrtf(ref_norm) * sqrtf(test_norm));
        } else if (!(ref_norm < 1e-9f && test_norm < 1e-9f)) {
            cosine_sim = 0.0f;
        }

        std::cout << "Comparison Results for " << test_name << ":" << std::endl;
        std::cout << "  Max Absolute Error: " << max_abs_error << std::endl;
        std::cout << "  Max Relative Error (for ref > 1e-4): " << max_rel_error * 100.0f << "%" << std::endl;
        std::cout << "  Avg Absolute Error: " << avg_abs_error << std::endl;
        std::cout << "  Cosine Similarity:  " << cosine_sim << std::endl;

        // ============================================================================
        // 最终的、更鲁棒的验证标准
        // ============================================================================

        // 1. 宏观上必须高度相似
        EXPECT_GT(cosine_sim, 0.999f) << test_name << " has poor cosine similarity";

        // 2. 检查不满足联合容差 (atol+rtol) 的离群点数量
        const float atol = 1e-3f; // 绝对容差
        const float rtol = 1e-2f; // 相对容差 (1%)
        int outlier_count = 0;

        for (size_t i = 0; i < reference.size(); ++i) {
            float ref_val = __half2float(reference[i]);
            float test_val = __half2float(test_result[i]);
            float tolerance = atol + rtol * std::abs(ref_val);

            if (std::abs(ref_val - test_val) > tolerance) {
                if (outlier_count < 5) { // 只打印前5个详细错误
                    std::cerr << "Outlier at index " << i << ": "
                              << "Got " << test_val << ", expected " << ref_val
                              << " (abs_err=" << std::abs(ref_val - test_val) << ", tol=" << tolerance << ")" << std::endl;
                }
                outlier_count++;
            }
        }
        EXPECT_LT(outlier_count, reference.size() * 0.025) // 允许最多 2.5% 的离群点
            << "Too many outliers (" << outlier_count << "/" << reference.size() 
            << ") outside combined tolerance (atol=" << atol << ", rtol=" << rtol << ")";

        if (cosine_sim > 0.999f && outlier_count < reference.size() * 0.025) {
            std::cout << "\n✓ " << test_name << " verification PASSED (within tolerance)" << std::endl;
        } else {
            std::cout << "\n✗ " << test_name << " verification FAILED" << std::endl;
        }
    }
    void CompareResultsBF16(
        const std::vector<unsigned short>& reference,
        const std::vector<unsigned short>& test_result, 
        const std::string& test_name
    ) {
        std::cout << "Comparing " << test_name << " results (BF16)..." << std::endl;
        
        ASSERT_EQ(reference.size(), test_result.size());
        if (reference.empty()) {
            std::cout << "Comparison skipped for empty vectors." << std::endl;
            return;
        }
    
        float max_abs_error = 0.0f;
        float max_rel_error = 0.0f; // Will only be calculated for non-tiny reference values
        float total_abs_error = 0.0f;
        float ref_norm = 0.0f, test_norm = 0.0f, dot_product = 0.0f;
    
        for (size_t i = 0; i < reference.size(); i++) {
            float ref_val = Bfloat16ToFloat(reference[i]);
            float test_val = Bfloat16ToFloat(test_result[i]);
            float abs_error = std::abs(ref_val - test_val);
            total_abs_error += abs_error;
            max_abs_error = std::max(max_abs_error, abs_error);
            
            if (std::abs(ref_val) > 1e-3f) { // Guard for BF16 is slightly different
                max_rel_error = std::max(max_rel_error, abs_error / std::abs(ref_val));
            }
        
            dot_product += ref_val * test_val;
            ref_norm += ref_val * ref_val;
            test_norm += test_val * test_val;
        }
    
        float avg_abs_error = total_abs_error / reference.size();
        float cosine_sim = 1.0f;
        if (ref_norm > 1e-9f && test_norm > 1e-9f) {
            cosine_sim = dot_product / (sqrtf(ref_norm) * sqrtf(test_norm));
        } else if (!(ref_norm < 1e-9f && test_norm < 1e-9f)) {
            cosine_sim = 0.0f;
        }
    
        std::cout << "Comparison Results for " << test_name << ":" << std::endl;
        std::cout << "  Max Absolute Error: " << max_abs_error << std::endl;
        std::cout << "  Max Relative Error (for ref > 1e-3): " << max_rel_error * 100.0f << "%" << std::endl;
        std::cout << "  Avg Absolute Error: " << avg_abs_error << std::endl;
        std::cout << "  Cosine Similarity:  " << cosine_sim << std::endl;
    
        // ============================================================================
        // 最终的、针对 BF16 的更鲁棒的验证标准
        // ============================================================================
        
        // 1. 宏观上必须高度相似 (BF16 容差比 FP16 稍宽松)
        EXPECT_GT(cosine_sim, 0.99f) << test_name << " has poor cosine similarity";
    
        // 2. 检查不满足联合容差 (atol+rtol) 的离群点数量
        const float atol = 5e-3f; // 绝对容差 (BF16 精度较低，容差更大)
        const float rtol = 5e-2f; // 相对容差 (5%)
        int outlier_count = 0;
    
        for (size_t i = 0; i < reference.size(); ++i) {
            float ref_val = Bfloat16ToFloat(reference[i]);
            float test_val = Bfloat16ToFloat(test_result[i]);
            float tolerance = atol + rtol * std::abs(ref_val);
            
            if (std::abs(ref_val - test_val) > tolerance) {
                if (outlier_count < 5) { // 只打印前5个详细错误
                    std::cerr << "Outlier at index " << i << ": "
                              << "Got " << test_val << ", expected " << ref_val
                              << " (abs_err=" << std::abs(ref_val - test_val) << ", tol=" << tolerance << ")" << std::endl;
                }
                outlier_count++;
            }
        }
        EXPECT_LT(outlier_count, reference.size() * 0.03) // 允许最多 3% 的离群点
            << "Too many outliers (" << outlier_count << "/" << reference.size() 
            << ") outside combined tolerance (atol=" << atol << ", rtol=" << rtol << ")";
    
        if (cosine_sim > 0.99f && outlier_count < reference.size() * 0.03) {
            std::cout << "\n✓ " << test_name << " verification PASSED (within tolerance)" << std::endl;
        } else {
            std::cout << "\n✗ " << test_name << " verification FAILED" << std::endl;
        }
    }
};


// ============================================================================
// 测试用例
// ============================================================================

// 测试1: 独立测试GPU反量化函数
TEST_F(MoEMxFp4Test, Step1_GPUDequantFunctionCorrectness) {
    TestGPUDequantFunctionCorrectness();
}

// 测试2: 独立测试量化往返精度
TEST_F(MoEMxFp4Test, Step2_QuantizationRoundTripAccuracy) {
    TestQuantizationRoundTripAccuracy();
}

TEST_F(MoEMxFp4Test, Step1_GPUDequantFunctionCorrectness_BF16) {
    TestGPUDequantFunctionCorrectnessBF16();
}

TEST_F(MoEMxFp4Test, Step2_QuantizationRoundTripAccuracy_BF16) {
    TestQuantizationRoundTripAccuracyBF16();
}
// 测试3: 独立测试激活函数
TEST_F(MoEMxFp4Test, Step3_ActivationFunctionCorrectness) {
    TestActivationFunctionCorrectness();
}

// 测试4: 完整FFN流程测试 - GELU (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_GELU_FP16_16x64x128) {
    MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, false);
}

TEST_F(MoEMxFp4Test, CompleteFFN_GELU_FP16_16x64x128_Random) {
    MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}
// 测试5: 完整FFN流程测试 - Swish (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_Swish_FP16_16x64x128) {
    MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeFp16, ActivationType::kSwish);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试6: 完整FFN流程测试 - ReLU (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_ReLU_FP16_32x64x256) {
    MoEStage2Config config(32, 64, 256, 1, DataType::kDataTypeFp16, ActivationType::kReLU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试7: 大规模测试 (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_GELU_FP16_64x128x512_Random) {
    MoEStage2Config config(64, 128, 512, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}
// 测试8: BF16数据类型测试
TEST_F(MoEMxFp4Test, CompleteFFN_GELU_BF16_32x64x128) {
    MoEStage2Config config(32, 64, 128, 1, DataType::kDataTypeBf16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

TEST_F(MoEMxFp4Test, CompleteFFN_Swish_BF16_16x64x128) {
    MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeBf16, ActivationType::kSwish);
    TestCompleteFFNPipeline(config, 1.0f, true);
}
// 测试9: 不同global_scale测试 (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_FP16_DifferentGlobalScales) {
    std::vector<float> test_scales = {0.5f, 1.0f, 2.0f, 4.0f};
    
    for (float scale : test_scales) {
        std::cout << "\n=== Testing FP16 with global_scale = " << scale << " ===" << std::endl;
        MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
        TestCompleteFFNPipeline(config, scale, true);
    }
}
// 测试10: 不同global_scale测试 (BF16)
TEST_F(MoEMxFp4Test, CompleteFFN_BF16_DifferentGlobalScales) {
    std::vector<float> test_scales = {0.5f, 1.0f, 2.0f};
    
    for (float scale : test_scales) {
        std::cout << "\n=== Testing BF16 with global_scale = " << scale << " ===" << std::endl;
        MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeBf16, ActivationType::kGELU);
        TestCompleteFFNPipeline(config, scale, true);
    }
}

// 测试11: 不同激活函数对比测试 (FP16)
TEST_F(MoEMxFp4Test, CompleteFFN_FP16_ActivationComparison_32x64x128) {
    std::vector<ActivationType> activations = {
        ActivationType::kGELU, 
        ActivationType::kSwish, 
        ActivationType::kReLU
    };
    
    for (auto activation : activations) {
        std::cout << "\n=== Testing FP16 with activation = " << static_cast<int>(activation) << " ===" << std::endl;
        MoEStage2Config config(32, 64, 128, 1, DataType::kDataTypeFp16, activation);
        TestCompleteFFNPipeline(config, 1.0f, true);
    }
}

// 测试12: 不同激活函数对比测试 (BF16)
TEST_F(MoEMxFp4Test, CompleteFFN_BF16_ActivationComparison_16x64x128) {
    std::vector<ActivationType> activations = {
        ActivationType::kGELU, 
        ActivationType::kSwish, 
        ActivationType::kReLU
    };
    
    for (auto activation : activations) {
        std::cout << "\n=== Testing BF16 with activation = " << static_cast<int>(activation) << " ===" << std::endl;
        MoEStage2Config config(16, 64, 128, 1, DataType::kDataTypeBf16, activation);
        TestCompleteFFNPipeline(config, 1.0f, true);
    }
}
// 测试13: 边界条件测试 - 最小尺寸
TEST_F(MoEMxFp4Test, CompleteFFN_FP16_MinimalSize_32x32x64) {
    MoEStage2Config config(4, 32, 64, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试14: 边界条件测试 - 中等尺寸
TEST_F(MoEMxFp4Test, CompleteFFN_FP16_MediumSize_128x96x192) {
    MoEStage2Config config(128, 96, 192, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试15: 维度对齐测试 - 确保所有维度都是32的倍数
TEST_F(MoEMxFp4Test, CompleteFFN_DimensionAlignment_64x160x320) {
    // 160 = 32 * 5, 320 = 32 * 10
    MoEStage2Config config(64, 160, 320, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试16: 性能压力测试 - 大规模配置
TEST_F(MoEMxFp4Test, CompleteFFN_StressTest_512x256x512) {
    MoEStage2Config config(512, 256, 512, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config, 1.0f, true);
}

// 测试17: 精度对比测试 - FP16 vs BF16
TEST_F(MoEMxFp4Test, CompleteFFN_PrecisionComparison_FP16_vs_BF16) {
    const unsigned tokens = 32, hidden = 64, intermediate = 128;
    
    std::cout << "\n=== FP16 vs BF16 Precision Comparison ===" << std::endl;
    
    // FP16测试
    std::cout << "\n--- Testing FP16 ---" << std::endl;
    MoEStage2Config config_fp16(tokens, hidden, intermediate, 1, DataType::kDataTypeFp16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config_fp16, 1.0f, true);
    
    // BF16测试
    std::cout << "\n--- Testing BF16 ---" << std::endl;
    MoEStage2Config config_bf16(tokens, hidden, intermediate, 1, DataType::kDataTypeBf16, ActivationType::kGELU);
    TestCompleteFFNPipeline(config_bf16, 1.0f, true);
}

// 测试18: 错误处理测试 - 无效配置
TEST_F(MoEMxFp4Test, ErrorHandling_InvalidDimensions) {
    // 测试不对齐的维度（不是32的倍数）
    MoEStage2Config config(16, 63, 128, 1, DataType::kDataTypeFp16, ActivationType::kGELU); // hidden_size=63不是32的倍数
    
    void *d_dummy;
    CheckHIPStatus(hipMalloc(&d_dummy, 1024));
    
    int result = MoECompleteFFNStage2(
        d_dummy, d_dummy, d_dummy, d_dummy, nullptr, 
        d_dummy, d_dummy, reinterpret_cast<float*>(d_dummy),
        config, nullptr
    );
    
    // 应该返回错误码
    EXPECT_NE(result, 0) << "Should fail with invalid dimensions";
    EXPECT_EQ(result, -1) << "Should return dimension error code";
    
    CheckHIPStatus(hipFree(d_dummy));
}

// 测试19: 错误处理测试 - 零维度
TEST_F(MoEMxFp4Test, ErrorHandling_ZeroDimensions) {
    MoEStage2Config config(0, 64, 128, 1, DataType::kDataTypeFp16, ActivationType::kGELU); // total_tokens=0
    
    void *d_dummy;
    CheckHIPStatus(hipMalloc(&d_dummy, 1024));
    
    int result = MoECompleteFFNStage2(
        d_dummy, d_dummy, d_dummy, d_dummy, nullptr, 
        d_dummy, d_dummy, reinterpret_cast<float*>(d_dummy),
        config, nullptr
    );
    
    // 应该成功但不做任何事情
    EXPECT_EQ(result, 0) << "Should succeed with zero tokens";
    
    CheckHIPStatus(hipFree(d_dummy));
}

}//

