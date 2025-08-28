// lib/gemm/rocm/quantization/moe/moe_benchmark.h
#pragma once

#include "moe_gemm_fp4.h"
#include "moe_test_utils.h"
#include <hip/hip_runtime.h>
#include "moe_test_utils.h"  // 因为使用了QuantizedFFNWeights

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// 高性能benchmark专用结构
// ============================================================================

/**
 * 预分配的GPU内存资源
 */
struct MoEGPUMemoryPool {
    void* d_input;
    void* d_output; 
    void* d_w1_weights;
    void* d_w2_weights;
    void* d_w1_scales;
    void* d_w2_scales;
    void* d_expert_indices;
    void* d_global_scale;
    
    bool allocated;
    
    MoEGPUMemoryPool() : allocated(false) {}
    ~MoEGPUMemoryPool() { Cleanup(); }
    
    void Cleanup();
};

/**
 * 轻量级benchmark配置
 */
struct FastBenchmarkConfig {
    MoEStage2Config moe_config;
    unsigned warmup_iterations;
    unsigned benchmark_iterations;
    bool use_hip_graph;          // 是否使用HIPGraph优化
    
    FastBenchmarkConfig() 
        : warmup_iterations(5), benchmark_iterations(100), 
          use_hip_graph(false) {}
};

/**
 * 精确的性能测量结果
 */
struct PreciseBenchmarkResult {
    double kernel_time_ms;       // 纯kernel执行时间
    double setup_time_ms;        // 设置时间
    double total_time_ms;        // 总时间
    double throughput_tflops;    // 吞吐量(TFLOPS)
    double memory_bandwidth;     // 内存带宽(GB/s)
    
    // 详细统计
    double min_time_ms;
    double max_time_ms;
    double std_deviation_ms;
    
    // Graph相关
    bool used_hip_graph;
    double graph_creation_time_ms;
};

// ============================================================================
// 快速benchmark类 - 解决计时问题
// ============================================================================

class FastMoEBenchmark {
public:
    explicit FastMoEBenchmark(const MoEStage2Config& config);
    ~FastMoEBenchmark();
    
    /**
     * 一次性设置：分配内存、准备数据
     */
    int Setup();
    
    /**
     * 纯kernel性能测量 - 解决685ms计时问题
     */
    int MeasureKernelPerformance(
        unsigned iterations,
        PreciseBenchmarkResult* result
    );
    
    /**
     * 使用HIPGraph的性能测量（可选）
     */
    int MeasureWithHIPGraph(
        unsigned iterations,
        PreciseBenchmarkResult* result
    );
    
    /**
     * 清理资源
     */
    void Cleanup();

private:
    MoEStage2Config config_;
    MoEGPUMemoryPool memory_pool_;
    QuantizedFFNWeights test_weights_;
    std::vector<half> test_input_fp16_;
    std::vector<unsigned short> test_input_bf16_;
    std::vector<unsigned> expert_indices_;
    float global_scale_;
    
    hipEvent_t start_event_, stop_event_;
    hipGraph_t hip_graph_;
    hipGraphExec_t graph_exec_;
    bool graph_created_;
    
    /**
     * 准备测试数据
     */
    int PrepareTestData();
    
    /**
     * 分配GPU内存
     */
    int AllocateGPUMemory();
    
    /**
     * 创建HIPGraph
     */
    int CreateHIPGraph();
};

// ============================================================================
// 便捷接口 - 兼容现有代码
// ============================================================================

/**
 * 快速benchmark接口 - 替代RunMoEBenchmark
 * 解决计时不准确问题
 */
int RunFastMoEBenchmark(
    const FastBenchmarkConfig& config,
    PreciseBenchmarkResult* result
);

/**
 * 简化接口 - 用于benchmark框架集成
 */
int MeasureMoEKernelPerformance(
    const MoEStage2Config& moe_config,
    unsigned iterations,
    double* avg_time_ms,
    double* throughput_tflops = nullptr
);

} // namespace causalflow::petit::rocm::quantization::moe