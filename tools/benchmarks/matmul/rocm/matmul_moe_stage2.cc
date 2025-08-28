// tools/benchmarks/matmul/rocm/matmul_moe_stage2.cc
#include "../matmul.h"
#include "gemm/rocm/quantization/moe/moe_benchmark.h"
#include "gemm/rocm/quantization/moe/moe_test_utils.h"  // 需要添加
#include "utils/hip_helper.h"
#include <memory>  
#include <sstream>  // for std::istringstream, std::ostringstream

namespace causalflow::petit::benchmark::matmul {

namespace rocm {

using namespace causalflow::petit::rocm::quantization::moe;

class MoEStage2Matmul : public Matmul {
public:
    MoEStage2Matmul(int m, int n, int k, DataType dtype)
        : m_(m), n_(n), k_(k), dtype_(dtype), benchmark_setup_(false) {
        
        printf("=== MoEStage2Matmul Constructor ===\n");
        printf("Received: m=%d, n=%d, k=%d\n", m, n, k);
        
        // 初始化默认配置
        InitializeConfig();
        
        printf("Mapped to MoE: tokens=%u, hidden=%u, intermediate=%u\n", 
               config_.total_tokens, config_.hidden_size, config_.intermediate_size);
        
        // 创建快速benchmark实例
        fast_benchmark_ = std::make_unique<FastMoEBenchmark>(config_);
    }

    absl::Status PrepareForBatchExecution(void *output, const void *input, 
                                         const void *, const void *,
                                         long, long, long, int batch_count) override {
        printf("=== PrepareForBatchExecution ===\n");
        printf("batch_count=%d\n", batch_count);
        
        if (batch_count != 1) {
            return absl::InvalidArgumentError("MoE only supports batch_count=1");
        }
        
        // 一次性设置benchmark资源
        if (!benchmark_setup_) {
            int ret = fast_benchmark_->Setup();
            if (ret != 0) {
                return absl::InternalError("Failed to setup benchmark resources");
            }
            benchmark_setup_ = true;
            printf("Benchmark resources allocated successfully\n");
        }
        
        return absl::OkStatus();
    }

    absl::Status SetAlgorithm(AlgorithmDescriptor algo) override {
        return absl::OkStatus();
    }

    absl::Status Execute(size_t repeat) override {
        printf("=== Execute Called ===\n");
        printf("repeat=%zu\n", repeat);
        
        if (!benchmark_setup_) {
            return absl::InternalError("Benchmark not initialized");
        }
        
        // 使用快速benchmark进行精确计时
        PreciseBenchmarkResult result;
        int ret = fast_benchmark_->MeasureKernelPerformance(
            static_cast<unsigned>(repeat), &result
        );
        
        if (ret != 0) {
            return absl::InternalError("Benchmark measurement failed");
        }
        
        // 输出详细性能信息
        printf("=== Precise Benchmark Results ===\n");
        printf("  Kernel time: %.3f ms (pure kernel execution)\n", result.kernel_time_ms);
        printf("  Setup time: %.3f ms\n", result.setup_time_ms);  
        printf("  Total time: %.3f ms\n", result.total_time_ms);
        printf("  Throughput: %.3f TFLOPS\n", result.throughput_tflops);
        printf("  Memory BW: %.3f GB/s\n", result.memory_bandwidth);
        printf("  Min/Max: %.3f/%.3f ms\n", result.min_time_ms, result.max_time_ms);
        
        return absl::OkStatus();
    }

    size_t GetAlgorithmCount() const override { return 1; }
    
    std::string GetAlgorithmRepr(size_t index) const override {
        return (index == 0) ? "fast_moe_stage2_kernel" : "";
    }

    /**
     * 根据命令行参数调整配置
     */
    void SetBenchmarkConfig(const std::string& config_str) {
        printf("Setting benchmark config: %s\n", config_str.c_str());
        
        // 解析配置字符串，例如: "activation=swish,scale=2.0,random=true"
        std::istringstream iss(config_str);
        std::string token;
        
        while (std::getline(iss, token, ',')) {
            size_t eq_pos = token.find('=');
            if (eq_pos != std::string::npos) {
                std::string key = token.substr(0, eq_pos);
                std::string value = token.substr(eq_pos + 1);
                
                ApplyConfigOption(key, value);
            }
        }
        
        // 重新创建benchmark实例
        fast_benchmark_ = std::make_unique<FastMoEBenchmark>(config_);
    }
    
    /**
     * 获取当前配置信息
     */
    std::string GetConfigInfo() const {
        std::ostringstream oss;
        oss << "MoE Config: " << m_ << "x" << k_ << "x" << n_ 
            << ", activation=" << static_cast<int>(config_.activation)
            << ", dtype=" << (dtype_ == DataType::kFp16 ? "fp16" : "bf16");
        return oss.str();
    }

private:
    // 保留命令行参数
    int m_, n_, k_;
    DataType dtype_;
    MoEStage2Config config_;
    std::unique_ptr<FastMoEBenchmark> fast_benchmark_;
    bool benchmark_setup_;
    
    void InitializeConfig() {
        // 基本参数映射
        config_.total_tokens = m_;        // tokens = m
        config_.hidden_size = k_;         // hidden = k  
        config_.intermediate_size = n_;   // intermediate = n
        config_.num_experts = 1;
        config_.input_type = (dtype_ == DataType::kFp16) ? 
            causalflow::petit::rocm::quantization::DataType::kDataTypeFp16 :
            causalflow::petit::rocm::quantization::DataType::kDataTypeBf16;
        config_.output_type = config_.input_type;
        config_.activation = ActivationType::kGELU;  // 默认激活函数
    }
    
    void ApplyConfigOption(const std::string& key, const std::string& value) {
        if (key == "activation") {
            if (value == "gelu") config_.activation = ActivationType::kGELU;
            else if (value == "swish") config_.activation = ActivationType::kSwish;
            else if (value == "relu") config_.activation = ActivationType::kReLU;
            else if (value == "identity") config_.activation = ActivationType::kIdentity;
            printf("Set activation to %s\n", value.c_str());
        }
        else if (key == "tokens") {
            m_ = std::stoi(value);
            config_.total_tokens = m_;
            printf("Set tokens to %d\n", m_);
        }
        else if (key == "hidden") {
            k_ = std::stoi(value);
            config_.hidden_size = k_;
            printf("Set hidden_size to %d\n", k_);
        }
        else if (key == "intermediate") {
            n_ = std::stoi(value);
            config_.intermediate_size = n_;
            printf("Set intermediate_size to %d\n", n_);
        }
        else if (key == "dtype") {
            if (value == "fp16") {
                dtype_ = DataType::kFp16;
                config_.input_type = causalflow::petit::rocm::quantization::DataType::kDataTypeFp16;
            } else if (value == "bf16") {
                dtype_ = DataType::kBf16;
                config_.input_type = causalflow::petit::rocm::quantization::DataType::kDataTypeBf16;
            }
            config_.output_type = config_.input_type;
            printf("Set dtype to %s\n", value.c_str());
        }
        else {
            printf("Unknown config option: %s=%s\n", key.c_str(), value.c_str());
        }
    }
};

class MoEStage2MatmulFactory : public MatmulFactory {
public:
    const char *GetPlatformName() const override { return "rocm"; }
    
    absl::Status CreateMatmul(hal::Device *dev, Matmul::DataType a_type,
                             Matmul::DataType c_type, int m, int n, int k,
                             std::unique_ptr<Matmul> *result) override {
        printf("=== CreateMatmul Called ===\n");
        printf("Parameters: m=%d, n=%d, k=%d\n", m, n, k);
        
        if (a_type != c_type) {
            return absl::InvalidArgumentError("Input/output types must match");
        }
        
        // 检查维度约束（可选）
        if (k % 32 != 0 || n % 32 != 0) {
            printf("Warning: Dimensions not aligned to 32, performance may be suboptimal\n");
        }
        
        *result = std::make_unique<MoEStage2Matmul>(m, n, k, a_type);
        printf("MoEStage2Matmul created successfully\n");
        return absl::OkStatus();
    }
};

} // namespace rocm

std::unique_ptr<MatmulFactory> CreateMatmulFactoryMoEStage2Backend() {
    printf("=== CreateMatmulFactoryMoEStage2Backend called ===\n");
    return std::make_unique<rocm::MoEStage2MatmulFactory>();
}

} // namespace causalflow::petit::benchmark::matmul