// lib/gemm/rocm/quantization/moe/moe_benchmark_interface.h  
// 为将来的benchmark系统预留的轻量接口文件

#pragma once

#include "moe_gemm_fp4.h"

namespace causalflow::petit::benchmark::matmul {

// ============================================================================
// MoE Benchmark接口预留 - 轻量版本
// ============================================================================

/**
 * 创建MoE benchmark实例 - 预留接口
 * 将来需要时再实现具体的MoEMatmulFactory
 */
// std::unique_ptr<MatmulFactory> CreateMoEMatmulFactory();  // 注释掉，将来实现

/**
 * MoE性能测试运行器 - 预留接口
 */
class MoEBenchmarkRunner {
public:
    // 构造函数预留
    // explicit MoEBenchmarkRunner(const rocm::quantization::moe::MoEBenchmarkConfig& config);
    
    // 接口预留，将来实现
    // rocm::quantization::moe::MoEBenchmarkResult RunBenchmark();
    // void PrintReport(const rocm::quantization::moe::MoEBenchmarkResult& result);
};

} // namespace causalflow::petit::benchmark::matmul

// ============================================================================
// 当前建议：先专注于核心功能测试
// Benchmark集成等生产环境特性可以在核心功能稳定后再添加
// ============================================================================

/*
使用路线图：

1. 当前阶段：
   - 专注于MoE核心算子的正确性
   - 使用现有的测试框架验证功能
   - 按专家建议的"dequant + hipBLASLt"参考实现

2. 下个阶段（核心功能稳定后）：
   - 实现benchmark集成
   - 添加性能优化
   - 扩展到多专家支持

3. 生产阶段：
   - 完整的错误处理和边界情况处理
   - 内存优化和kernel选择策略
   - 与现有系统的深度集成
*/