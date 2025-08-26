// lib/gemm/rocm/quantization/moe/moe_gemm_fp4.h
#pragma once

#include "../types.h"
#include "../gemm.h"
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// 激活函数类型定义
// ============================================================================

enum class ActivationType {
    kGELU = 0,      // Gaussian Error Linear Unit
    kSwish = 1,     // Swish activation (SiLU)
    kReLU = 2,      // Rectified Linear Unit
    kIdentity = -1  // 无激活函数（仅用于测试）
};

// ============================================================================
// MoE Stage2 完整FFN配置结构
// ============================================================================

struct MoEStage2Config {
    unsigned total_tokens;      // 总token数量
    unsigned hidden_size;       // 隐藏层维度  
    unsigned intermediate_size; // 专家网络中间层维度（FFN维度）
    unsigned num_experts;       // 专家数量
    DataType input_type;        // 输入数据类型 (FP16/BF16)
    DataType output_type;       // 输出数据类型 (FP16/BF16) 
    ActivationType activation;  // 激活函数类型
    
    // 默认构造函数
    MoEStage2Config() = default;
    
    // 便捷构造函数
    MoEStage2Config(unsigned tokens, unsigned hidden, unsigned intermediate, 
                    unsigned experts = 1, DataType dtype = DataType::kDataTypeFp16,
                    ActivationType act = ActivationType::kGELU)
        : total_tokens(tokens), hidden_size(hidden), intermediate_size(intermediate),
          num_experts(experts), input_type(dtype), output_type(dtype),
          activation(act) {}
};

// ============================================================================
// 主要接口：完整的MoE Stage2 FFN
// ============================================================================

/**
 * 完整的MoE Stage2 FFN计算：包含两层线性变换 + 激活函数
 * 
 * 执行完整的FFN流程：
 * 1. H = input @ W1        (第一层线性变换)
 * 2. H_act = Activation(H) (激活函数：GELU/Swish/ReLU)
 * 3. Y = H_act @ W2        (第二层线性变换)
 * 
 * @param final_output         最终输出 [total_tokens, hidden_size]
 * @param input                输入token特征 [total_tokens, hidden_size]
 * @param w1_weights           第一层量化权重 [num_experts, hidden_size, intermediate_size_compressed]
 * @param w2_weights           第二层量化权重 [num_experts, intermediate_size, hidden_size_compressed]
 * @param expert_indices       token到expert的映射 [total_tokens]
 * @param w1_scales            W1的MxFP4 scale参数 [num_experts, hidden_size/32, intermediate_size]
 * @param w2_scales            W2的MxFP4 scale参数 [num_experts, intermediate_size/32, hidden_size]
 * @param global_scale         全局缩放因子
 * @param config               MoE配置参数
 * @param stream               HIP流（可选，默认nullptr使用默认流）
 * 
 * @return 0表示成功，负值表示错误码
 */
int MoECompleteFFNStage2(
    void* final_output,                      
    const void* input,                       
    const void* w1_weights,                  
    const void* w2_weights,                  
    const unsigned* expert_indices,          
    const void* w1_scales,                   
    const void* w2_scales,                   
    const float* global_scale,
    const MoEStage2Config& config,
    hipStream_t stream = nullptr
);

// ============================================================================
// 测试专用接口 - 用于单元测试和验证
// ============================================================================

/**
 * 测试GPU反量化函数的正确性
 */
int CallTestGPUDequantKernel(
    float* gpu_results,                   // 输出：GPU反量化结果
    const unsigned* fp4_values,          // 输入：4-bit值数组
    const float* scales,                  // 输入：scale数组
    unsigned num_tests,                   // 测试数量
    hipStream_t stream = nullptr,          // HIP流
    DataType element_type = DataType::kDataTypeFp16 
);

/**
 * 完整反量化kernel的Host接口函数
 */
int CallFullDequantMxFp4Kernel(
    void* dequant_weights,                // 输出：反量化权重
    const void* quant_weights,            // 输入：量化权重
    const void* scales,                   // 输入：缩放因子
    unsigned hidden_size,
    unsigned intermediate_size,
    DataType element_type,                // 元素类型
    hipStream_t stream = nullptr          // HIP流
);

/**
 * 测试激活函数的正确性
 */
int CallTestActivationKernel(
    void* activated_output,               // 输出：激活后结果
    const void* input,                    // 输入：待激活数据
    unsigned total_elements,              // 元素总数
    int activation_type,                  // 激活函数类型
    DataType element_type,                // 元素数据类型
    hipStream_t stream = nullptr          // HIP流
);

// ============================================================================
// Benchmark接口
// ============================================================================

struct MoEBenchmarkConfig {
    MoEStage2Config moe_config;
    unsigned warmup_iterations;    // 预热迭代次数
    unsigned benchmark_iterations; // 性能测试迭代次数
    bool enable_profiling;         // 是否启用profiling
    
    MoEBenchmarkConfig() 
        : warmup_iterations(5), benchmark_iterations(100), 
          enable_profiling(false) {}
};

struct MoEBenchmarkResult {
    double avg_time_ms;        // 平均执行时间(毫秒)
    double min_time_ms;        // 最小执行时间
    double max_time_ms;        // 最大执行时间  
    double throughput_tflops;  // 吞吐量(TFLOPS)
    double memory_bandwidth;   // 内存带宽利用率
    double layer1_time_ms;     // 第一层时间
    double activation_time_ms; // 激活函数时间
    double layer2_time_ms;     // 第二层时间
};

int RunMoEBenchmark(
    const MoEBenchmarkConfig& config,
    MoEBenchmarkResult* result
);

// ============================================================================ 
// 错误码定义
// ============================================================================

enum class MoEError {
    Success = 0,
    InvalidConfig = -1,
    UnsupportedDataType = -2,
    InsufficientMemory = -3, 
    KernelLaunchFailed = -4,
    InvalidExpertIndices = -5,
    InvalidDimensions = -6,
    UnsupportedActivation = -7,
    SharedMemoryError = -8
};

/**
 * 将错误码转换为可读字符串
 */
const char* MoEErrorToString(MoEError error);

/**
 * 激活函数类型转换辅助函数
 */
inline int ActivationTypeToInt(ActivationType activation) {
    return static_cast<int>(activation);
}

inline ActivationType IntToActivationType(int activation) {
    return static_cast<ActivationType>(activation);
}

} // namespace causalflow::petit::rocm::quantization::moe