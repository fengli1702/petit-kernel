// lib/gemm/rocm/quantization/moe/mxfp4_types.h
#pragma once

#include "../types.h"
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization {

// ============================================================================
// MXFP4 E2M1 格式定义 (Microscaling Format, 4-bit):
// - 1位符号 (Sign)  
// - 2位指数 (Exponent)
// - 1位尾数 (Mantissa)
// - 使用共享的scale因子 (shared scale per block)
// ============================================================================

/**
 * MXFP4量化参数结构
 */
struct MxFp4QuantParams {
    unsigned sign_bits;         // 符号位数 (通常为1)
    unsigned exponent_bits;     // 指数位数 (E2M1中为2)  
    unsigned mantissa_bits;     // 尾数位数 (E2M1中为1)
    unsigned scale_bits;        // 共享缩放的位数 (通常为32位float)
    unsigned block_size;        // 共享缩放的块大小 (通常为32)

    /**
     * 获取E2M1格式的默认参数
     */
    static constexpr MxFp4QuantParams E2M1_Default() {
        return {
            .sign_bits = 1,      // 1位符号
            .exponent_bits = 2,  // 2位指数  
            .mantissa_bits = 1,  // 1位尾数
            .scale_bits = 32,    // 32位float scale
            .block_size = 32     // 32个元素共享一个scale
        };
    }
};

// ============================================================================
// MXFP4反量化常量定义
// ============================================================================

/// 每个量化块包含的元素数量 (32个4-bit值共享一个scale)
static constexpr unsigned kMxFp4BlockSize = 32;

/// 32个4-bit值 (128 bits) 需要4个32-bit整数来存储
static constexpr unsigned kMxFp4PackedDwords = 4;

/// 共享缩放因子的位数 (使用float32)
static constexpr unsigned kMxFp4ScaleBits = 32;

/// E2M1格式的值域 (无符号部分: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
static constexpr unsigned kMxFp4E2M1ValueRange = 8;

/// E2M1格式的最大绝对值
static constexpr float kMxFp4E2M1MaxValue = 6.0f;

// ============================================================================
// 内存布局相关常量
// ============================================================================

/**
 * MxFP4权重在内存中的布局格式
 */
enum class MxFp4MemoryLayout {
    kRowMajor,          // 行主序：[input_dim, output_dim]
    kColumnMajor,       // 列主序：[output_dim, input_dim] 
    kPacked             // 压缩格式：8个4-bit值打包成一个uint32
};

/**
 * MxFP4量化权重的元数据
 */
struct MxFp4WeightMetadata {
    unsigned rows;                    // 权重矩阵行数 (input_dim)
    unsigned cols;                    // 权重矩阵列数 (output_dim)
    unsigned num_blocks_per_row;      // 每行的量化块数量
    unsigned packed_size_bytes;       // 压缩后的权重数据大小(字节)
    unsigned scales_size_bytes;       // scale数据大小(字节)
    MxFp4MemoryLayout layout;         // 内存布局格式
    DataType original_dtype;          // 原始数据类型 (FP16/BF16)
    
    /**
     * 根据矩阵维度计算元数据
     */
    static MxFp4WeightMetadata CalculateMetadata(
        unsigned input_dim, unsigned output_dim, DataType dtype, 
        MxFp4MemoryLayout layout = MxFp4MemoryLayout::kRowMajor) {
        
        MxFp4WeightMetadata meta;
        meta.rows = input_dim;
        meta.cols = output_dim; 
        meta.num_blocks_per_row = (input_dim + kMxFp4BlockSize - 1) / kMxFp4BlockSize;
        // 8个4-bit值打包成1个uint32
        meta.packed_size_bytes = ((input_dim * output_dim + 7) / 8) * sizeof(unsigned);  
        meta.scales_size_bytes = meta.num_blocks_per_row * output_dim * sizeof(float);
        meta.layout = layout;
        meta.original_dtype = dtype;
        return meta;
    }
    
    /**
     * 验证维度对齐
     */
    bool IsValidDimensions() const {
        return (rows % kMxFp4BlockSize == 0) && (rows > 0) && (cols > 0);
    }
    
    /**
     * 计算总内存需求
     */
    size_t GetTotalMemoryBytes() const {
        return packed_size_bytes + scales_size_bytes;
    }
};

// ============================================================================  
// GPU kernel使用的常量
// ============================================================================

/// GPU warp大小 (AMD GPU)
static constexpr unsigned kGpuWarpSize = 64;  

/// 每个thread处理的4-bit值数量 (优化内存访问)
static constexpr unsigned kValuesPerThread = 8;

/// 每个warp处理的4-bit值数量  
static constexpr unsigned kValuesPerWarp = kGpuWarpSize * kValuesPerThread;

/// 推荐的thread block大小
static constexpr unsigned kRecommendedBlockSize = 256;

/// 推荐的shared memory大小限制 (字节)
static constexpr size_t kMaxSharedMemoryBytes = 48 * 1024;  // 48KB

// ============================================================================
// 实用工具宏和内联函数
// ============================================================================

/**
 * 计算给定元素数量需要的量化块数
 */
__host__ __device__ constexpr unsigned 
CalculateNumBlocks(unsigned num_elements) {
    return (num_elements + kMxFp4BlockSize - 1) / kMxFp4BlockSize;
}

/**
 * 计算压缩后的数据大小 (以uint32为单位)
 */
__host__ __device__ constexpr unsigned 
CalculatePackedSizeDwords(unsigned num_elements) {
    return (num_elements + 7) / 8;  // 8个4-bit值打包成1个uint32
}

/**
 * 检查维度是否对齐到量化块大小
 */
__host__ __device__ constexpr bool
IsAlignedToBlockSize(unsigned dimension) {
    return (dimension % kMxFp4BlockSize) == 0;
}

/**
 * 计算给定配置所需的shared memory大小
 */
__host__ constexpr size_t
CalculateSharedMemorySize(unsigned intermediate_size, DataType dtype) {
    size_t elem_size = (dtype == DataType::kDataTypeFp16) ? sizeof(unsigned short) : sizeof(unsigned short);
    return intermediate_size * elem_size;
}

/**
 * 检查shared memory大小是否在限制内
 */
__host__ constexpr bool
IsSharedMemoryWithinLimit(size_t shared_mem_size) {
    return shared_mem_size <= kMaxSharedMemoryBytes;
}

/**
 * 计算最优block大小
 */
__host__ constexpr unsigned
CalculateOptimalBlockSize(unsigned total_tokens, unsigned max_dim) {
    // 简单策略：基于最大维度和token数量
    if (max_dim <= 128) return 128;
    else if (max_dim <= 256) return 256;
    else return kRecommendedBlockSize;
}

} // namespace causalflow::petit::rocm::quantization