#pragma once

#include "types.h"
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization {

// ============================================================================
// MXFP4 E2M1 格式 (Microscaling Format, 4-bit):
// - 1位符号 (Sign)
// - 2位指数 (Exponent)
// - 1位尾数 (Mantissa)
// - 使用共享的8位指数缩放因子 (shared 8-bit exponent scale)
// ============================================================================

// MXFP4 量化参数
struct MxFp4QuantParams {
    unsigned sign_bits;         // 符号位数
    unsigned exponent_bits;     // 指数位数
    unsigned mantissa_bits;     // 尾数位数
    unsigned scale_bits;        // 共享缩放的位数
    unsigned block_size;        // 共享缩放的块大小

    static constexpr MxFp4QuantParams E2M1_Default() {
        return {1, 2, 1, 8, 32}; // 1-sign, 2-exp, 1-mantissa, 8-bit scale, 32-element block
    }
};


// MxFP4 反量化常量
// 每个块包含32个元素
static constexpr unsigned kMxFp4BlockSize = 32;

// 32个4-bit值 (128 bits) 需要 4个32-bit整数来存储
static constexpr unsigned kMxFp4PackedDwords = 4;

// 共享缩放因子是8位
static constexpr unsigned kMxFp4ScaleBits = 8;


} // namespace causalflow::petit::rocm::quantization
