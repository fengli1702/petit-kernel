// lib/gemm/rocm/quantization/moe/moe_benchmark.cc
#include "moe_benchmark.h"
#include "moe_test_utils.h"
#include "utils/hip_helper.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace causalflow::petit::rocm::quantization::moe {

// ============================================================================
// MoEGPUMemoryPool Implementation
// ============================================================================

void MoEGPUMemoryPool::Cleanup() {
    if (!allocated) return;
    
    if (d_input) CheckHIPStatus(hipFree(d_input));
    if (d_output) CheckHIPStatus(hipFree(d_output));
    if (d_w1_weights) CheckHIPStatus(hipFree(d_w1_weights));
    if (d_w2_weights) CheckHIPStatus(hipFree(d_w2_weights));
    if (d_w1_scales) CheckHIPStatus(hipFree(d_w1_scales));
    if (d_w2_scales) CheckHIPStatus(hipFree(d_w2_scales));
    if (d_expert_indices) CheckHIPStatus(hipFree(d_expert_indices));
    if (d_global_scale) CheckHIPStatus(hipFree(d_global_scale));
    
    allocated = false;
}

// ============================================================================
// FastMoEBenchmark Implementation
// ============================================================================

FastMoEBenchmark::FastMoEBenchmark(const MoEStage2Config& config)
    : config_(config), global_scale_(1.0f), graph_created_(false) {
    
    CheckHIPStatus(hipEventCreate(&start_event_));
    CheckHIPStatus(hipEventCreate(&stop_event_));
    
    hip_graph_ = nullptr;
    graph_exec_ = nullptr;
}

FastMoEBenchmark::~FastMoEBenchmark() {
    Cleanup();
}

int FastMoEBenchmark::Setup() {
    printf("FastMoEBenchmark::Setup - Preparing resources...\n");
    
    // 1. 准备测试数据
    int ret = PrepareTestData();
    if (ret != 0) {
        printf("Failed to prepare test data: %d\n", ret);
        return ret;
    }
    
    // 2. 分配GPU内存
    ret = AllocateGPUMemory();
    if (ret != 0) {
        printf("Failed to allocate GPU memory: %d\n", ret);
        return ret;
    }
    
    printf("FastMoEBenchmark setup completed successfully\n");
    return 0;
}

int FastMoEBenchmark::PrepareTestData() {
    printf("Preparing test data for config: %ux%ux%u\n", 
           config_.total_tokens, config_.hidden_size, config_.intermediate_size);
    
    // 使用moe_test_utils中的共用数据准备函数
    if (config_.input_type == DataType::kDataTypeFp16) {
        auto [input_data, ffn_weights, expert_indices] = 
            fp16_data::PrepareFFNTestData(config_, true); // use_random_data=true
        
        test_input_fp16_ = std::move(input_data);
        test_weights_ = std::move(ffn_weights);
        expert_indices_ = std::move(expert_indices);
    } else if (config_.input_type == DataType::kDataTypeBf16) {
        auto [input_data, ffn_weights, expert_indices] = 
            bf16_data::PrepareFFNTestData(config_, true);
        
        test_input_bf16_ = std::move(input_data);
        test_weights_ = std::move(ffn_weights);
        expert_indices_ = std::move(expert_indices);
    } else {
        printf("Unsupported data type\n");
        return -1;
    }
    
    printf("Test data prepared: input_size=%zu, w1_data=%zu, w2_data=%zu\n",
           config_.input_type == DataType::kDataTypeFp16 ? test_input_fp16_.size() : test_input_bf16_.size(),
           test_weights_.w1_data.size(), test_weights_.w2_data.size());
    
    return 0;
}

int FastMoEBenchmark::AllocateGPUMemory() {
    printf("Allocating GPU memory...\n");
    
    size_t elem_size = (config_.input_type == DataType::kDataTypeFp16) ? 
                       sizeof(half) : sizeof(unsigned short);
    
    // 计算内存大小
    size_t input_size = config_.total_tokens * config_.hidden_size * elem_size;
    size_t output_size = config_.total_tokens * config_.hidden_size * elem_size;
    size_t w1_size = test_weights_.w1_data.size() * sizeof(unsigned);
    size_t w2_size = test_weights_.w2_data.size() * sizeof(unsigned);
    size_t w1_scales_size = test_weights_.w1_scales.size() * sizeof(float);
    size_t w2_scales_size = test_weights_.w2_scales.size() * sizeof(float);
    size_t indices_size = expert_indices_.size() * sizeof(unsigned);
    
    printf("Memory allocation sizes:\n");
    printf("  Input: %zu bytes\n", input_size);
    printf("  Output: %zu bytes\n", output_size);
    printf("  W1 weights: %zu bytes, W1 scales: %zu bytes\n", w1_size, w1_scales_size);
    printf("  W2 weights: %zu bytes, W2 scales: %zu bytes\n", w2_size, w2_scales_size);
    
    // 分配GPU内存
    CheckHIPStatus(hipMalloc(&memory_pool_.d_input, input_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_output, output_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_w1_weights, w1_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_w2_weights, w2_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_w1_scales, w1_scales_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_w2_scales, w2_scales_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_expert_indices, indices_size));
    CheckHIPStatus(hipMalloc(&memory_pool_.d_global_scale, sizeof(float)));
    
    memory_pool_.allocated = true;
    
    // 拷贝数据到GPU (一次性完成)
    printf("Copying data to GPU...\n");
    
    if (config_.input_type == DataType::kDataTypeFp16) {
        CheckHIPStatus(hipMemcpy(memory_pool_.d_input, test_input_fp16_.data(), 
                                input_size, hipMemcpyHostToDevice));
    } else {
        CheckHIPStatus(hipMemcpy(memory_pool_.d_input, test_input_bf16_.data(), 
                                input_size, hipMemcpyHostToDevice));
    }
    
    CheckHIPStatus(hipMemcpy(memory_pool_.d_w1_weights, test_weights_.w1_data.data(), 
                            w1_size, hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(memory_pool_.d_w2_weights, test_weights_.w2_data.data(), 
                            w2_size, hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(memory_pool_.d_w1_scales, test_weights_.w1_scales.data(), 
                            w1_scales_size, hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(memory_pool_.d_w2_scales, test_weights_.w2_scales.data(), 
                            w2_scales_size, hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(memory_pool_.d_expert_indices, expert_indices_.data(), 
                            indices_size, hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(memory_pool_.d_global_scale, &global_scale_, 
                            sizeof(float), hipMemcpyHostToDevice));
    
    printf("GPU memory setup completed\n");
    return 0;
}

int FastMoEBenchmark::MeasureKernelPerformance(
    unsigned iterations,
    PreciseBenchmarkResult* result) {
    
    if (!memory_pool_.allocated) {
        printf("Error: GPU memory not allocated\n");
        return -1;
    }
    
    printf("Starting precise kernel measurement for %u iterations...\n", iterations);
    
    // 预热运行
    const unsigned warmup_iters = std::min(5U, iterations / 10);
    printf("Warming up with %u iterations...\n", warmup_iters);
    
    for (unsigned i = 0; i < warmup_iters; ++i) {
        int ret = MoECompleteFFNStage2(
            memory_pool_.d_output,
            memory_pool_.d_input,
            memory_pool_.d_w1_weights,
            memory_pool_.d_w2_weights,
            reinterpret_cast<const unsigned*>(memory_pool_.d_expert_indices),
            memory_pool_.d_w1_scales,
            memory_pool_.d_w2_scales,
            reinterpret_cast<const float*>(memory_pool_.d_global_scale),
            config_,
            nullptr
        );
        if (ret != 0) {
            printf("Warmup kernel failed: %d\n", ret);
            return ret;
        }
    }
    CheckHIPStatus(hipDeviceSynchronize());
    
    // 精确计时：多次测量
    std::vector<float> kernel_times;
    kernel_times.reserve(iterations);
    
    printf("Running %u timed iterations...\n", iterations);
    
    auto setup_start = std::chrono::high_resolution_clock::now();
    
    for (unsigned i = 0; i < iterations; ++i) {
        // 单次kernel计时
        CheckHIPStatus(hipEventRecord(start_event_));
        
        int ret = MoECompleteFFNStage2(
            memory_pool_.d_output,
            memory_pool_.d_input,
            memory_pool_.d_w1_weights,
            memory_pool_.d_w2_weights,
            reinterpret_cast<const unsigned*>(memory_pool_.d_expert_indices),
            memory_pool_.d_w1_scales,
            memory_pool_.d_w2_scales,
            reinterpret_cast<const float*>(memory_pool_.d_global_scale),
            config_,
            nullptr
        );
        
        CheckHIPStatus(hipEventRecord(stop_event_));
        CheckHIPStatus(hipEventSynchronize(stop_event_));
        
        if (ret != 0) {
            printf("Kernel failed at iteration %u: %d\n", i, ret);
            return ret;
        }
        
        float elapsed_ms;
        CheckHIPStatus(hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
        kernel_times.push_back(elapsed_ms);
        
        // 每50次迭代输出进度
        if ((i + 1) % 50 == 0 || i == 0) {
            printf("  Iteration %u/%u: %.3f ms\n", i + 1, iterations, elapsed_ms);
        }
    }
    
    auto setup_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(setup_end - setup_start).count();
    
    // 统计分析
    double sum = std::accumulate(kernel_times.begin(), kernel_times.end(), 0.0);
    double avg_time = sum / kernel_times.size();
    double min_time = *std::min_element(kernel_times.begin(), kernel_times.end());
    double max_time = *std::max_element(kernel_times.begin(), kernel_times.end());
    
    // 计算标准差
    double variance = 0.0;
    for (float time : kernel_times) {
        variance += (time - avg_time) * (time - avg_time);
    }
    double std_dev = std::sqrt(variance / kernel_times.size());
    
    // 计算吞吐量 (TFLOPS)
    uint64_t ops_per_run = 4ULL * config_.total_tokens * config_.hidden_size * config_.intermediate_size;
    double avg_time_sec = avg_time / 1000.0;
    double throughput_tflops = (ops_per_run / avg_time_sec) / 1e12;
    
    // 估算内存带宽
    uint64_t bytes_per_run = config_.total_tokens * config_.hidden_size * sizeof(half) * 3; // 简化估算
    double memory_bandwidth = (bytes_per_run / avg_time_sec) / 1e9; // GB/s
    
    // 填充结果
    result->kernel_time_ms = avg_time;
    result->setup_time_ms = total_time_ms - sum; // 设置开销
    result->total_time_ms = total_time_ms;
    result->throughput_tflops = throughput_tflops;
    result->memory_bandwidth = memory_bandwidth;
    result->min_time_ms = min_time;
    result->max_time_ms = max_time;
    result->std_deviation_ms = std_dev;
    result->used_hip_graph = false;
    result->graph_creation_time_ms = 0.0;
    
    printf("=== Precise Timing Results ===\n");
    printf("Kernel time: %.3f ± %.3f ms (min: %.3f, max: %.3f)\n", 
           avg_time, std_dev, min_time, max_time);
    printf("Setup overhead: %.3f ms\n", result->setup_time_ms);
    printf("Total time: %.3f ms\n", total_time_ms);
    printf("Throughput: %.3f TFLOPS\n", throughput_tflops);
    printf("Memory BW: %.3f GB/s\n", memory_bandwidth);
    
    return 0;
}

int FastMoEBenchmark::CreateHIPGraph() {
    if (graph_created_ || !memory_pool_.allocated) {
        return -1;
    }
    
    printf("Creating HIP Graph for optimized execution...\n");
    
    auto graph_start = std::chrono::high_resolution_clock::now();
    
    // 开始图捕获
    CheckHIPStatus(hipStreamBeginCapture(nullptr, hipStreamCaptureModeGlobal));
    
    // 捕获多次kernel调用以增加图的大小和精度
    const unsigned graph_iterations = 10; // 在图中包含10次kernel调用
    for (unsigned i = 0; i < graph_iterations; ++i) {
        int ret = MoECompleteFFNStage2(
            memory_pool_.d_output,
            memory_pool_.d_input,
            memory_pool_.d_w1_weights,
            memory_pool_.d_w2_weights,
            reinterpret_cast<const unsigned*>(memory_pool_.d_expert_indices),
            memory_pool_.d_w1_scales,
            memory_pool_.d_w2_scales,
            reinterpret_cast<const float*>(memory_pool_.d_global_scale),
            config_,
            nullptr
        );
        if (ret != 0) {
            printf("Failed to capture kernel in graph: %d\n", ret);
            hipError_t capture_result = hipStreamEndCapture(nullptr, &hip_graph_);
            if (capture_result != hipSuccess) {
                printf("Failed to end stream capture: %d\n", capture_result);
                return -1;
            }
            return ret;
        }
    }
    
    // 结束捕获
    CheckHIPStatus(hipStreamEndCapture(nullptr, &hip_graph_));
    
    // 实例化图
    CheckHIPStatus(hipGraphInstantiate(&graph_exec_, hip_graph_, nullptr, nullptr, 0));
    
    auto graph_end = std::chrono::high_resolution_clock::now();
    double graph_creation_time = std::chrono::duration<double, std::milli>(graph_end - graph_start).count();
    
    graph_created_ = true;
    printf("HIP Graph created successfully in %.3f ms (contains %u kernel calls)\n", 
           graph_creation_time, graph_iterations);
    
    return 0;
}

int FastMoEBenchmark::MeasureWithHIPGraph(
    unsigned iterations,
    PreciseBenchmarkResult* result) {
    
    if (!graph_created_) {
        int ret = CreateHIPGraph();
        if (ret != 0) {
            printf("Failed to create HIP Graph, falling back to normal measurement\n");
            return MeasureKernelPerformance(iterations, result);
        }
    }
    
    printf("Starting HIP Graph measurement for %u iterations...\n", iterations);
    
    // 预热
    for (int i = 0; i < 3; ++i) {
        CheckHIPStatus(hipGraphLaunch(graph_exec_, nullptr));
        CheckHIPStatus(hipDeviceSynchronize());
    }
    
    // 精确计时
    std::vector<float> graph_times;
    const unsigned graph_batches = (iterations + 9) / 10; // 每个graph包含10次kernel调用
    graph_times.reserve(graph_batches);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (unsigned i = 0; i < graph_batches; ++i) {
        CheckHIPStatus(hipEventRecord(start_event_));
        CheckHIPStatus(hipGraphLaunch(graph_exec_, nullptr));
        CheckHIPStatus(hipEventRecord(stop_event_));
        CheckHIPStatus(hipEventSynchronize(stop_event_));
        
        float elapsed_ms;
        CheckHIPStatus(hipEventElapsedTime(&elapsed_ms, start_event_, stop_event_));
        graph_times.push_back(elapsed_ms);
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    // 计算单次kernel平均时间（每个graph包含10次调用）
    double total_graph_time = std::accumulate(graph_times.begin(), graph_times.end(), 0.0);
    double avg_kernel_time = total_graph_time / (graph_batches * 10); // 除以总kernel调用次数
    
    // 计算吞吐量
    uint64_t ops_per_run = 4ULL * config_.total_tokens * config_.hidden_size * config_.intermediate_size;
    double avg_time_sec = avg_kernel_time / 1000.0;
    double throughput_tflops = (ops_per_run / avg_time_sec) / 1e12;
    
    // 填充结果
    result->kernel_time_ms = avg_kernel_time;
    result->setup_time_ms = 0.0;
    result->total_time_ms = total_time_ms;
    result->throughput_tflops = throughput_tflops;
    result->memory_bandwidth = 0.0; // 简化
    result->min_time_ms = *std::min_element(graph_times.begin(), graph_times.end()) / 10;
    result->max_time_ms = *std::max_element(graph_times.begin(), graph_times.end()) / 10;
    result->std_deviation_ms = 0.0;
    result->used_hip_graph = true;
    result->graph_creation_time_ms = 0.0; // 已创建
    
    printf("=== HIP Graph Results ===\n");
    printf("Average kernel time: %.3f ms (from %u graph executions)\n", 
           avg_kernel_time, graph_batches);
    printf("Throughput: %.3f TFLOPS\n", throughput_tflops);
    
    return 0;
}

void FastMoEBenchmark::Cleanup() {
    if (start_event_) CheckHIPStatus(hipEventDestroy(start_event_));
    if (stop_event_) CheckHIPStatus(hipEventDestroy(stop_event_));
    
    if (graph_exec_) CheckHIPStatus(hipGraphExecDestroy(graph_exec_));
    if (hip_graph_) CheckHIPStatus(hipGraphDestroy(hip_graph_));
    
    memory_pool_.Cleanup();
}

// ============================================================================
// 便捷接口实现
// ============================================================================

int RunFastMoEBenchmark(
    const FastBenchmarkConfig& config,
    PreciseBenchmarkResult* result) {
    
    FastMoEBenchmark benchmark(config.moe_config);
    
    int ret = benchmark.Setup();
    if (ret != 0) {
        return ret;
    }
    
    if (config.use_hip_graph) {
        return benchmark.MeasureWithHIPGraph(config.benchmark_iterations, result);
    } else {
        return benchmark.MeasureKernelPerformance(config.benchmark_iterations, result);
    }
}

int MeasureMoEKernelPerformance(
    const MoEStage2Config& moe_config,
    unsigned iterations,
    double* avg_time_ms,
    double* throughput_tflops) {
    
    FastBenchmarkConfig config;
    config.moe_config = moe_config;
    config.benchmark_iterations = iterations;
    config.use_hip_graph = false;
    
    PreciseBenchmarkResult result;
    int ret = RunFastMoEBenchmark(config, &result);
    
    if (ret == 0) {
        *avg_time_ms = result.kernel_time_ms;
        if (throughput_tflops) {
            *throughput_tflops = result.throughput_tflops;
        }
    }
    
    return ret;
}

} // namespace causalflow::petit::rocm::quantization::moe