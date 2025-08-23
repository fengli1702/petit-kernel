#include "gemm_moe_fp4.h"
#include "../mxfp4_dequant.cuh"
#include "../mxfp4_types.h"

#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization::fp4 {

// 基础的MxFP4 MOE kernel - 使用官方E2M1格式
template <typename ElementType>
__global__ void MoeMxFp4Kernel(
    ElementType* __restrict__ output,              // [total_tokens, intermediate_size]
    const ElementType* __restrict__ input,         // [total_tokens, hidden_size] 
    const unsigned* __restrict__ weights,          // [num_experts, hidden_size, intermediate_size_compressed]
    const unsigned* __restrict__ expert_indices,   // [total_tokens]
    const float* __restrict__ scales,              // float类型
    const float* __restrict__ global_scale,
    const unsigned total_tokens,
    const unsigned hidden_size,
    const unsigned intermediate_size
) {
    const unsigned token_id = blockIdx.x;
    const unsigned out_dim = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (token_id >= total_tokens || out_dim >= intermediate_size) return;
    
    // 获取当前token对应的expert索引
    unsigned expert_id = expert_indices[token_id];
    
    // 计算数据偏移
    const ElementType* token_input = input + token_id * hidden_size;
    ElementType* token_output = output + token_id * intermediate_size;
    
    // 计算权重和scale的偏移
    const unsigned weights_per_expert = hidden_size * intermediate_size / 8;  // 8个4bit值打包成1个uint32
    const unsigned* expert_weights = weights + expert_id * weights_per_expert;
    
    // scale现在是float类型
    const unsigned scales_per_expert = (hidden_size / kMxFp4BlockSize) * intermediate_size;
    const float* expert_scales = scales + expert_id * scales_per_expert;
    
    float accumulator = 0.0f;
    
    // 执行矩阵乘法：output[out_dim] = sum(input[h] * weight[h][out_dim])
    for (unsigned h = 0; h < hidden_size; h++) {
        // 加载输入值
        float input_val;
        if constexpr (std::is_same_v<ElementType, half>) {
            input_val = __half2float(token_input[h]);
        } else {
            input_val = __bfloat162float(token_input[h]);
        }
        
        // 列主序索引计算
        const unsigned packed_cols_stride = hidden_size / 8;
        unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
        unsigned bit_offset = (h % 8) * 4;
        
        // 加载packed权重
        unsigned packed_weight = expert_weights[packed_idx];
        unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
        
        // 计算scale索引（每32个hidden维度共享一个scale）
        unsigned h_block = h / kMxFp4BlockSize;
        unsigned scale_idx = h_block * intermediate_size + out_dim;
        float scale = expert_scales[scale_idx];
        
        // 使用官方E2M1反量化函数
        float weight_val = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
        
        // 累加
        accumulator += input_val * weight_val;
    }
    
    // 应用全局scale并写回结果
    accumulator *= global_scale[0];
    
    if constexpr (std::is_same_v<ElementType, half>) {
        token_output[out_dim] = __float2half(accumulator);
    } else {
        token_output[out_dim] = __float2bfloat16(accumulator);
    }
}

// MxFP4 MOE kernel shared memory
template <typename ElementType>
__global__ void MoeMxFp4OptimizedKernel(
    ElementType* __restrict__ output,
    const ElementType* __restrict__ input,
    const unsigned* __restrict__ weights,
    const unsigned* __restrict__ expert_indices,
    const float* __restrict__ scales,
    const float* __restrict__ global_scale,
    const unsigned total_tokens,
    const unsigned hidden_size,
    const unsigned intermediate_size
) {
    extern __shared__ char shared_mem[];
    unsigned* shared_expert_id = reinterpret_cast<unsigned*>(shared_mem);
    ElementType* shared_input = reinterpret_cast<ElementType*>(shared_mem + sizeof(unsigned));
    
    const unsigned token_id = blockIdx.x;
    const unsigned out_dim = threadIdx.x + blockIdx.y * blockDim.x;
    
    // --- Step 1: 加载expert_id ---
    if (threadIdx.x == 0) {
        if (token_id < total_tokens) {
            shared_expert_id[0] = expert_indices[token_id];
        } else {
            shared_expert_id[0] = 0; 
        }
    }
    __syncthreads();
    
    const unsigned expert_id = shared_expert_id[0];
    
    // --- Step 2: 边界检查---
    bool valid_thread = (token_id < total_tokens && out_dim < intermediate_size);
    
    // --- Step 3: 协作加载输入向量到共享内存 ---
    const ElementType* token_input = (token_id < total_tokens) ? 
        (input + token_id * hidden_size) : nullptr;
    
    for (unsigned i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        if (token_input != nullptr) {
            shared_input[i] = token_input[i];
        } else {
            // 无效线程加载零值，虽然结果不会被使用
            shared_input[i] = ElementType(0);
        }
    }
    __syncthreads();
    
    // --- Step 4: 只有有效线程进行计算 ---
    if (!valid_thread) return;
    
    // 计算权重和scale偏移
    const unsigned weights_per_expert = hidden_size * intermediate_size / 8;
    const unsigned* expert_weights = weights + expert_id * weights_per_expert;
    
    const unsigned scales_per_expert = (hidden_size / kMxFp4BlockSize) * intermediate_size;
    const float* expert_scales = scales + expert_id * scales_per_expert;
    
    float accumulator = 0.0f;
    const unsigned packed_cols_stride = hidden_size / 8;
    
    // 使用shared memory中的输入数据进行计算
    for (unsigned h = 0; h < hidden_size; h++) {
        float input_val;
        if constexpr (std::is_same_v<ElementType, half>) {
            input_val = __half2float(shared_input[h]);
        } else {
            input_val = __bfloat162float(shared_input[h]);
        }
        
        // 权重解包逻辑保持不变
        unsigned packed_idx = out_dim * packed_cols_stride + (h / 8);
        unsigned bit_offset = (h % 8) * 4;
        
        unsigned packed_weight = expert_weights[packed_idx];
        unsigned fp4_val = (packed_weight >> bit_offset) & 0xF;
        
        unsigned h_block = h / kMxFp4BlockSize;
        unsigned scale_idx = h_block * intermediate_size + out_dim;
        float scale = expert_scales[scale_idx];
        
        float weight_val = SimpleMxFp4DequantE2M1<ElementType>(fp4_val, scale);
        accumulator += input_val * weight_val;
    }
    
    // 应用全局scale并写回结果
    accumulator *= global_scale[0];
    
    ElementType* token_output = output + token_id * intermediate_size;
    if constexpr (std::is_same_v<ElementType, half>) {
        token_output[out_dim] = __float2half(accumulator);
    } else {
        token_output[out_dim] = __float2bfloat16(accumulator);
    }
}
// Host接口实现

// Host接口实现 - 修复版本
int MoeMxFp4SecondStage(
    unsigned *experts_output,
    const unsigned *gating_output,
    const unsigned *expert_weights,
    const unsigned *expert_indices,
    const float *scales,
    const float *global_scale,
    const unsigned total_tokens,
    const unsigned hidden_size,
    const unsigned intermediate_size,
    const PetitSolutionHints &hints,
    hipStream_t stream) {
    
    if (total_tokens == 0 || hidden_size == 0 || intermediate_size == 0) {
        return 0;  // 空输入，直接返回成功
    }
    
    // 参数有效性检查
    if (hidden_size % kMxFp4BlockSize != 0) {
        return -1;  // hidden_size必须是32的倍数
    }
    
    // 计算grid和block配置
    const unsigned threads_per_block = 256;
    const unsigned blocks_per_token = (intermediate_size + threads_per_block - 1) / threads_per_block;
    
    dim3 blockDim(threads_per_block);
    dim3 gridDim(total_tokens, blocks_per_token);
    
    bool use_optimized = (hidden_size >= 64 && total_tokens >= 4);
    //use_optimized = false;
    
    if (hints.a_type == DataType::kDataTypeFp16) {
        if (use_optimized) {
            size_t shared_mem_size = sizeof(unsigned) + hidden_size * sizeof(half);
            // 确保8字节对齐
            shared_mem_size = ((shared_mem_size + 7) / 8) * 8;
            
            MoeMxFp4OptimizedKernel<half><<<gridDim, blockDim, shared_mem_size, stream>>>(
                reinterpret_cast<half*>(experts_output),
                reinterpret_cast<const half*>(gating_output),
                expert_weights,
                expert_indices,
                scales,
                global_scale,
                total_tokens,
                hidden_size,
                intermediate_size
            );
        } else {
            MoeMxFp4Kernel<half><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<half*>(experts_output),
                reinterpret_cast<const half*>(gating_output),
                expert_weights,
                expert_indices,
                scales,
                global_scale,
                total_tokens,
                hidden_size,
                intermediate_size
            );
        }
    } else if (hints.a_type == DataType::kDataTypeBf16) {
        if (use_optimized) {
            size_t shared_mem_size = sizeof(unsigned) + hidden_size * sizeof(__hip_bfloat16);
            // 确保8字节对齐
            shared_mem_size = ((shared_mem_size + 7) / 8) * 8;

            MoeMxFp4OptimizedKernel<__hip_bfloat16><<<gridDim, blockDim, shared_mem_size, stream>>>(
                reinterpret_cast<__hip_bfloat16*>(experts_output),
                reinterpret_cast<const __hip_bfloat16*>(gating_output),
                expert_weights,
                expert_indices,
                scales,
                global_scale,
                total_tokens,
                hidden_size,
                intermediate_size
            );
        } else {
            MoeMxFp4Kernel<__hip_bfloat16><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<__hip_bfloat16*>(experts_output),
                reinterpret_cast<const __hip_bfloat16*>(gating_output),
                expert_weights,
                expert_indices,
                scales,
                global_scale,
                total_tokens,
                hidden_size,
                intermediate_size
            );
        }
    } else {
        return -3;  // 不支持的数据类型
    }
    
    // 检查kernel启动错误
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        return -4;  // kernel启动失败
    }
    
    return 0;  // 成功
    }
}