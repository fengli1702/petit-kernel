// lib/gemm/rocm/quantization/moe/moe_gemm_fp4_grid.cc
#include "moe_gemm_fp4.h"
#include "mxfp4_dequant.cuh"
#include "mxfp4_types.h"
#include "../gemm.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// 激活函数实现
// ============================================================================

template <typename ElementType>
__device__ __forceinline__ float GELU(float x) {
    // 近似 GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float a = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + a * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

template <typename ElementType>
__device__ __forceinline__ float Swish(float x, float beta = 1.0f) {
    return x / (1.0f + expf(-beta * x));
}

template <typename ElementType>
__device__ __forceinline__ float ReLU(float x) {
    return fmaxf(0.0f, x);
}

template <typename ElementType>
__device__ __forceinline__ float ApplyActivation(float x, int activation_type) {
    switch (activation_type) {
        case 0: return GELU<ElementType>(x);    // GELU
        case 1: return Swish<ElementType>(x);   // Swish
        case 2: return ReLU<ElementType>(x);    // ReLU
        default: return x;                      // Identity
    }
}

// ============================================================================
// 完整的MoE FFN Kernel - 单个kernel实现整个FFN流程
// ============================================================================

template <typename ElementType>
__global__ void MoECompleteFFNKernel(
    ElementType* __restrict__ final_output,        // [total_tokens, hidden_size]
    const ElementType* __restrict__ input,         // [total_tokens, hidden_size]
    const unsigned* __restrict__ w1_weights,       // [num_experts, hidden_size, intermediate_size_packed]
    const unsigned* __restrict__ w2_weights,       // [num_experts, intermediate_size, hidden_size_packed]
    const unsigned* __restrict__ expert_indices,   // [total_tokens]
    const float* __restrict__ w1_scales,           // [num_experts, hidden_size/32, intermediate_size]
    const float* __restrict__ w2_scales,           // [num_experts, intermediate_size/32, hidden_size]
    const float* __restrict__ global_scale,
    const unsigned total_tokens,
    const unsigned hidden_size,
    const unsigned intermediate_size,
    const int activation_type
) {
    // 每个 block 处理一个 token
    const unsigned token_id = blockIdx.x;
    const unsigned tid = threadIdx.x;
    
    if (token_id >= total_tokens) return;
    
    // 使用 shared memory 存储中间结果
    extern __shared__ char shared_mem[];
    ElementType* shared_intermediate = reinterpret_cast<ElementType*>(shared_mem);
    
    unsigned expert_id = expert_indices[token_id];
    const ElementType* token_input = input + token_id * hidden_size;
    ElementType* token_output = final_output + token_id * hidden_size;
    
    // ========== Step 1: 第一层线性变换 H = X @ W1 ==========
    const unsigned w1_weights_per_expert = hidden_size * intermediate_size / 8;
    const unsigned* expert_w1_weights = w1_weights + expert_id * w1_weights_per_expert;
    
    const unsigned w1_scales_per_expert = (hidden_size / kMxFp4BlockSize) * intermediate_size;
    const float* expert_w1_scales = w1_scales + expert_id * w1_scales_per_expert;
    
    // 每个线程计算多个输出维度
    for (unsigned out_dim = tid; out_dim < intermediate_size; out_dim += blockDim.x) {
        float accumulator = 0.0f;
        const unsigned packed_cols_stride = hidden_size / 8;
        
        for (unsigned h = 0; h < hidden_size; h++) {
            float input_val;
            if constexpr (std::is_same_v<ElementType, half>) {
                input_val = __half2float(token_input[h]);
            } else {
                input_val = __bfloat162float(token_input[h]);
            }
            
            // 解包权重
            unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
            unsigned bit_offset = (h % 8) * 4;
            unsigned packed_weight = expert_w1_weights[packed_idx];
            unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
            
            // 获取scale
            unsigned h_block = h / kMxFp4BlockSize;
            unsigned scale_idx = h_block * intermediate_size + out_dim;
            float scale = expert_w1_scales[scale_idx];
            
            // 反量化
            float weight_val = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
            accumulator += input_val * weight_val  ;
        }
        
        // 存储到shared memory
        if constexpr (std::is_same_v<ElementType, half>) {
            shared_intermediate[out_dim] = __float2half(accumulator);
        } else {
            shared_intermediate[out_dim] = __float2bfloat16(accumulator);
        }
    }
    
    __syncthreads();
    
    // ========== Step 2: 激活函数 H_act = Activation(H) ==========
    for (unsigned dim = tid; dim < intermediate_size; dim += blockDim.x) {
        float intermediate_val;
        if constexpr (std::is_same_v<ElementType, half>) {
            intermediate_val = __half2float(shared_intermediate[dim]);
        } else {
            intermediate_val = __bfloat162float(shared_intermediate[dim]);
        }
        
        float activated_val = ApplyActivation<ElementType>(intermediate_val, activation_type);
        
        if constexpr (std::is_same_v<ElementType, half>) {
            shared_intermediate[dim] = __float2half(activated_val);
        } else {
            shared_intermediate[dim] = __float2bfloat16(activated_val);
        }
    }
    
    __syncthreads();
    
    // ========== Step 3: 第二层线性变换 Y = H_act @ W2 ==========
    const unsigned w2_weights_per_expert = intermediate_size * hidden_size / 8;
    const unsigned* expert_w2_weights = w2_weights + expert_id * w2_weights_per_expert;
    
    const unsigned w2_scales_per_expert = (intermediate_size / kMxFp4BlockSize) * hidden_size;
    const float* expert_w2_scales = w2_scales + expert_id * w2_scales_per_expert;
    
    // 每个线程计算多个输出维度
    for (unsigned out_dim = tid; out_dim < hidden_size; out_dim += blockDim.x) {
        float accumulator = 0.0f;
        const unsigned packed_cols_stride = intermediate_size / 8;
        
        for (unsigned i = 0; i < intermediate_size; i++) {
            float activated_val;
            if constexpr (std::is_same_v<ElementType, half>) {
                activated_val = __half2float(shared_intermediate[i]);
            } else {
                activated_val = __bfloat162float(shared_intermediate[i]);
            }
            
            // 解包W2权重
            unsigned packed_idx = out_dim * packed_cols_stride + (i / 8);
            unsigned bit_offset = (i % 8) * 4;
            unsigned packed_weight = expert_w2_weights[packed_idx];
            unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
            
            // 获取scale
            unsigned i_block = i / kMxFp4BlockSize;
            unsigned scale_idx = i_block * hidden_size + out_dim;
            float scale = expert_w2_scales[scale_idx];
            
            // 反量化
            float weight_val = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
            accumulator += activated_val * weight_val;
        }
        
        // 应用全局scale并写入最终输出
        accumulator *= global_scale[0];
        
        if constexpr (std::is_same_v<ElementType, half>) {
            token_output[out_dim] = __float2half(accumulator);
        } else {
            token_output[out_dim] = __float2bfloat16(accumulator);
        }
    }
}

// ============================================================================
// 测试专用：独立的激活函数kernel
// ============================================================================

template <typename ElementType>
__global__ void ActivationTestKernel(
    ElementType* __restrict__ output,
    const ElementType* __restrict__ input,
    const unsigned total_elements,
    const int activation_type
) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total_elements) return;
    
    float input_val;
    if constexpr (std::is_same_v<ElementType, half>) {
        input_val = __half2float(input[idx]);
    } else {
        input_val = __bfloat162float(input[idx]);
    }
    
    float result = ApplyActivation<ElementType>(input_val, activation_type);
    
    if constexpr (std::is_same_v<ElementType, half>) {
        output[idx] = __float2half(result);
    } else {
        output[idx] = __float2bfloat16(result);
    }
}

// ============================================================================
// 测试专用：反量化函数测试kernel
// ============================================================================

template <typename ElementType>
__global__ void TestGPUDequantKernel(
    float* __restrict__ gpu_results,              
    const unsigned* __restrict__ fp4_values,     
    const float* __restrict__ scales,            
    unsigned num_tests                            
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;
    
    unsigned fp4_val = fp4_values[idx];
    float scale = scales[idx];
    
    gpu_results[idx] = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
}

// ============================================================================
// 测试专用：完整反量化kernel
// ============================================================================

template <typename ElementType>
__global__ void FullDequantMxFp4Kernel(
    ElementType* __restrict__ dequant_weights,    
    const unsigned* __restrict__ quant_weights,   
    const float* __restrict__ scales,             
    const unsigned hidden_size,
    const unsigned intermediate_size
) {
    const unsigned h = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned out_dim = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (h >= hidden_size || out_dim >= intermediate_size) return;
    
    // 解包量化权重
    const unsigned packed_cols_stride = hidden_size / 8;
    unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
    unsigned bit_offset = (h % 8) * 4;
    
    unsigned packed_weight = quant_weights[packed_idx];
    unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
    
    // 计算scale索引
    unsigned h_block = h / kMxFp4BlockSize;
    unsigned scale_idx = h_block * intermediate_size + out_dim;
    float scale = scales[scale_idx];
    
    // 反量化
    float weight_val = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
    
    // 写入结果 (行主序：h * intermediate_size + out_dim)
    if constexpr (std::is_same_v<ElementType, half>) {
        dequant_weights[h * intermediate_size + out_dim] = __float2half(weight_val);
    } else {
        dequant_weights[h * intermediate_size + out_dim] = __float2bfloat16(weight_val);
    }
}

// ============================================================================
// Host接口实现
// ============================================================================

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
    hipStream_t stream
) {
    if (config.total_tokens == 0 || config.hidden_size == 0 || config.intermediate_size == 0) {
        return 0;
    }
    
    if (config.hidden_size % kMxFp4BlockSize != 0 || config.intermediate_size % kMxFp4BlockSize != 0) {
        return -1; // 维度必须是32的倍数
    }
    
    // 计算shared memory大小 (存储intermediate结果)
    size_t elem_size = (config.input_type == DataType::kDataTypeFp16) ? sizeof(half) : sizeof(__hip_bfloat16);
    size_t shared_mem_size = config.intermediate_size * elem_size;
    
    // 每个block处理一个token，每个线程处理多个输出维度
    const unsigned threads_per_block = 256;
    dim3 blockDim(threads_per_block);
    dim3 gridDim(config.total_tokens);
    
    int activation_type = static_cast<int>(config.activation);
    
    if (config.input_type == DataType::kDataTypeFp16) {
        MoECompleteFFNKernel<half><<<gridDim, blockDim, shared_mem_size, stream>>>(
            reinterpret_cast<half*>(final_output),
            reinterpret_cast<const half*>(input),
            reinterpret_cast<const unsigned*>(w1_weights),
            reinterpret_cast<const unsigned*>(w2_weights),
            expert_indices,
            reinterpret_cast<const float*>(w1_scales),
            reinterpret_cast<const float*>(w2_scales),
            global_scale,
            config.total_tokens,
            config.hidden_size,
            config.intermediate_size,
            activation_type
        );
    } else if (config.input_type == DataType::kDataTypeBf16) {
        MoECompleteFFNKernel<__hip_bfloat16><<<gridDim, blockDim, shared_mem_size, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(final_output),
            reinterpret_cast<const __hip_bfloat16*>(input),
            reinterpret_cast<const unsigned*>(w1_weights),
            reinterpret_cast<const unsigned*>(w2_weights),
            expert_indices,
            reinterpret_cast<const float*>(w1_scales),
            reinterpret_cast<const float*>(w2_scales),
            global_scale,
            config.total_tokens,
            config.hidden_size,
            config.intermediate_size,
            activation_type
        );
    } else {
        return -2; // 不支持的数据类型
    }
    
    hipError_t err = hipGetLastError();
    return (err != hipSuccess) ? -3 : 0;
}

// ============================================================================
// 测试专用接口实现
// ============================================================================

int CallTestGPUDequantKernel(
    float* gpu_results,
    const unsigned* fp4_values,
    const float* scales,
    unsigned num_tests,
    hipStream_t stream,
    DataType element_type
) {
    dim3 blockDim(256);
    dim3 gridDim((num_tests + blockDim.x - 1) / blockDim.x);
    
    if (element_type == DataType::kDataTypeFp16) {
        TestGPUDequantKernel<half><<<gridDim, blockDim, 0, stream>>>(
            gpu_results, fp4_values, scales, num_tests);
    } else if (element_type == DataType::kDataTypeBf16) {
        TestGPUDequantKernel<__hip_bfloat16><<<gridDim, blockDim, 0, stream>>>(
            gpu_results, fp4_values, scales, num_tests);
    } else {
        return -2; // Unsupported type
    }
    
    hipError_t err = hipGetLastError();
    return (err != hipSuccess) ? -1 : 0;
}

int CallFullDequantMxFp4Kernel(
    void* dequant_weights,
    const void* quant_weights,
    const void* scales,
    unsigned hidden_size,
    unsigned intermediate_size,
    DataType element_type,
    hipStream_t stream
) {
    if (hidden_size % kMxFp4BlockSize != 0) {
        return -1;
    }
    
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (hidden_size + blockDim.x - 1) / blockDim.x,
        (intermediate_size + blockDim.y - 1) / blockDim.y
    );
    
    if (element_type == DataType::kDataTypeFp16) {
        FullDequantMxFp4Kernel<half><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<half*>(dequant_weights),
            reinterpret_cast<const unsigned*>(quant_weights),
            reinterpret_cast<const float*>(scales),
            hidden_size, intermediate_size
        );
    } else if (element_type == DataType::kDataTypeBf16) {
        FullDequantMxFp4Kernel<__hip_bfloat16><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(dequant_weights),
            reinterpret_cast<const unsigned*>(quant_weights),
            reinterpret_cast<const float*>(scales),
            hidden_size, intermediate_size
        );
    } else {
        return -2;
    }
    
    hipError_t err = hipGetLastError();
    return (err != hipSuccess) ? -3 : 0;
}

int CallTestActivationKernel(
    void* activated_output,
    const void* input,
    unsigned total_elements,
    int activation_type,
    DataType element_type,
    hipStream_t stream
) {
    dim3 blockDim(256);
    dim3 gridDim((total_elements + blockDim.x - 1) / blockDim.x);
    
    if (element_type == DataType::kDataTypeFp16) {
        ActivationTestKernel<half><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<half*>(activated_output),
            reinterpret_cast<const half*>(input),
            total_elements,
            activation_type
        );
    } else if (element_type == DataType::kDataTypeBf16) {
        ActivationTestKernel<__hip_bfloat16><<<gridDim, blockDim, 0, stream>>>(
            reinterpret_cast<__hip_bfloat16*>(activated_output),
            reinterpret_cast<const __hip_bfloat16*>(input),
            total_elements,
            activation_type
        );
    } else {
        return -2;
    }
    
    hipError_t err = hipGetLastError();
    return (err != hipSuccess) ? -3 : 0;
}

// ============================================================================
// 错误处理函数
// ============================================================================

const char* MoEErrorToString(MoEError error) {
    switch (error) {
    case MoEError::Success:
        return "Success";
    case MoEError::InvalidConfig:
        return "Invalid configuration";
    case MoEError::UnsupportedDataType:
        return "Unsupported data type";
    case MoEError::InsufficientMemory:
        return "Insufficient memory";
    case MoEError::KernelLaunchFailed:
        return "Kernel launch failed";
    case MoEError::InvalidExpertIndices:
        return "Invalid expert indices";
    case MoEError::InvalidDimensions:
        return "Invalid dimensions";
    case MoEError::UnsupportedActivation:
        return "Unsupported activation function";
    case MoEError::SharedMemoryError:
        return "Shared memory allocation error";
    default:
        return "Unknown error";
    }
}

// ============================================================================
// Benchmark预留函数 - 简单实现
// ============================================================================

int RunMoEBenchmark(
    const MoEBenchmarkConfig& config,
    MoEBenchmarkResult* result) {
    // TODO: 实现benchmark逻辑
    if (result) {
        result->avg_time_ms = 0.0;
        result->min_time_ms = 0.0;
        result->max_time_ms = 0.0;
        result->throughput_tflops = 0.0;
        result->memory_bandwidth = 0.0;
        result->layer1_time_ms = 0.0;
        result->activation_time_ms = 0.0;
        result->layer2_time_ms = 0.0;
    }
    return 0;
}

} // namespace causalflow::petit::rocm::quantization::moe