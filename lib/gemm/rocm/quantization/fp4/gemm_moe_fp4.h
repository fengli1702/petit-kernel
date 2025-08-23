#pragma once

#include "gemm/rocm/quantization/types.h"
#include "gemm/rocm/quantization/gemm.h"
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization::fp4 {

// MxFP4 MOE Phase 2: experts_output = gating_output @ expert_weights
// 使用官方MxFP4 E2M1格式
// 参数说明：
// - experts_output: [total_tokens, intermediate_size] 输出矩阵
// - gating_output: [total_tokens, hidden_size] 输入矩阵 (FP16/BF16)
// - expert_weights: [num_experts, hidden_size, intermediate_size_compressed] MxFP4量化权重
// - expert_indices: [total_tokens] 每个token对应的expert索引
// - scales: [num_experts, hidden_size/32, intermediate_size] MxFP4量化的scale参数 (float类型)
// - global_scale: 全局缩放因子
// - total_tokens: token总数
// - hidden_size: 隐藏层维度
// - intermediate_size: 专家网络中间层维度
// - hints: 数据类型提示
// - stream: HIP流
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
    hipStream_t stream = nullptr
);

} // namespace causalflow::petit::rocm::quantization::fp4