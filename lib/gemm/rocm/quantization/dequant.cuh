#pragma once

#include "gemm/rocm/amd_intrinsics.cuh"
#include "gemm/rocm/quantization/types.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <numeric>

namespace causalflow::petit::rocm::quantization {

// Fast dequantization from FP4 to FP16/BF16, adopted from
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/gptq_marlin/dequant.h
template <class TargetType, DataType kSrcType> struct Dequantizer;
template <class TargetType, bool kUpscale> struct DequantizerForFp8Scale;

// Upscale the scales by 2 ** 7 to the special E5M3 format. It avoids denorms
// and ensures the scale is always in bound when casts to half2 / bf16.
// Scaling by 2 ** 7 will make maximum of the exponent to 6 + 8 = 15.
static constexpr unsigned kFp8ScaleBias = 7;

namespace detail {

template <class TargetType, unsigned kSrcEx_, unsigned kFp16Ex_>
struct DequantizerToFp16Impl {
    static constexpr unsigned kSrcEx = kSrcEx_;
    static constexpr unsigned kFp16Ex = kFp16Ex_;
    static constexpr int kRightShift = kFp16Ex - kSrcEx;
    static constexpr int kExpOffset =
        2 * (1 << (kFp16Ex - 1)) - (1 << (kSrcEx - 1)) - 1;

    __device__ static void Dequant(TargetType *out, unsigned v) {
        static constexpr unsigned kMask = 0x70007000;
        static constexpr unsigned kSignMask = 0x80008000;
        unsigned *o_ptr = reinterpret_cast<unsigned *>(out);
        o_ptr[0] = (v & kSignMask) | ((v & kMask) >> kRightShift);
        v <<= 4;
        o_ptr[1] = (v & kSignMask) | ((v & kMask) >> kRightShift);
    }
};

template <class TargetType, unsigned kSrcEx_, unsigned kFp16Ex_, bool kUpscale_>
struct DequantizerForFp8ScaleImpl {
    static constexpr unsigned kSrcEx = kSrcEx_;
    static constexpr unsigned kFp16Ex = kFp16Ex_;
    static constexpr bool kUpscale = kUpscale_;
    static constexpr unsigned kScaleEx = 5;
    static constexpr unsigned kSrcBias = (1 << (kSrcEx - 1)) - 1;
    static constexpr unsigned kScaleBias = (1 << (kScaleEx - 1)) - 1;
    static constexpr unsigned kFp16Bias = (1 << (kFp16Ex - 1)) - 1;

    // The MatrixCore might flush all the denorms to zeros for bf16, therefore
    // we upscale the scales before the mfma instructions.
    static constexpr int kUpscaleExpBiasRaw =
        ((1 << kFp16Ex) - 1) - ((1 << kScaleEx) - 1);

    static constexpr unsigned kGSExpBias =
        kUpscale ? kFp16Bias - kSrcBias - (kUpscaleExpBiasRaw - kFp16Bias) -
                       kScaleBias - kFp8ScaleBias
                 : 0;
    static constexpr unsigned kFp32Ex = 8;
    static constexpr unsigned kFp32Bias = (1 << (kFp32Ex - 1)) - 1;
    static constexpr unsigned kDequantExpBiasU32 = (kGSExpBias + kFp32Bias)
                                                   << (32 - kFp32Ex - 1);

    static constexpr float GlobalScaleFactor() {
        return std::bit_cast<float>(kDequantExpBiasU32);
    }

    __device__ static unsigned AdjustPackedScaleBias(unsigned s) {
        // Divide 2 ** kFp8ScaleBias introduced in preprocessing
        static constexpr int kReversePreprocessBias =
            kFp16Bias - kScaleBias - kFp8ScaleBias;
        if constexpr (kUpscale || kFp16Ex == 8) {
            static constexpr unsigned kScaleBiasU16 =
                (kUpscale ? kUpscaleExpBiasRaw : kReversePreprocessBias)
                << (16 - kFp16Ex - 1);
            static constexpr unsigned kScaleBiasU32 =
                (kScaleBiasU16 << 16) | kScaleBiasU16;
            return s + kScaleBiasU32;
        } else {
            // In the float16 high precision mode, we unscale the kFp8ScaleBias
            // when dequantizing the qweights.
            return s;
        }
    }
};

} // namespace detail

template <>
struct Dequantizer<half2, kDataTypeFp4e2m1>
    : public detail::DequantizerToFp16Impl<half2, 2, 5> {
    using Element = half;
    using VectorType = half2;

    __device__ static Element Bias(bool high_precision) {
        const unsigned off =
            high_precision ? kExpOffset - kFp8ScaleBias : kExpOffset;
        unsigned short v = off << (15 - kFp16Ex);
        return *(const Element *)&v;
    }
};

template <>
struct Dequantizer<__hip_bfloat162, kDataTypeFp4e2m1>
    : public detail::DequantizerToFp16Impl<__hip_bfloat162, 2, 8> {
    using Element = __hip_bfloat16;
    using VectorType = __hip_bfloat162;
    __device__ static Element Bias(bool high_precision) {
        static constexpr unsigned short v = kExpOffset << (15 - kFp16Ex);
        return *(const Element *)&v;
    }
};

template <bool kUpscale_>
struct DequantizerForFp8Scale<half2, kUpscale_>
    : public detail::DequantizerForFp8ScaleImpl<half2, 2, 5, kUpscale_> {
    using Base = detail::DequantizerForFp8ScaleImpl<half2, 2, 5, kUpscale_>;

    __device__ static void Dequant(half2 *out, unsigned short s) {
        unsigned r = amdgcn_perm_b32(0, s, 0x0c010c00);
        reinterpret_cast<unsigned *>(out)[0] = r << 7;
    }

    __device__ static void DequantFullScale(half2 *out, unsigned short s) {
        unsigned v;
        Dequant(reinterpret_cast<half2 *>(&v), s);
        v = Base::AdjustPackedScaleBias(v);
        reinterpret_cast<unsigned *>(out)[0] = v;
    }
};

template <bool kUpscale_>
struct DequantizerForFp8Scale<__hip_bfloat162, kUpscale_>
    : public detail::DequantizerForFp8ScaleImpl<__hip_bfloat162, 2, 8,
                                                kUpscale_> {
    using Base =
        detail::DequantizerForFp8ScaleImpl<__hip_bfloat162, 2, 8, kUpscale_>;

    __device__ static void Dequant(__hip_bfloat162 *out, unsigned short s) {
        unsigned v = amdgcn_perm_b32(0, s, 0x0c010c00);
        reinterpret_cast<unsigned *>(out)[0] = v << 4;
    }

    __device__ static void DequantFullScale(__hip_bfloat162 *out,
                                            unsigned short s) {
        unsigned v;
        Dequant(reinterpret_cast<__hip_bfloat162 *>(&v), s);
        v = Base::AdjustPackedScaleBias(v);
        reinterpret_cast<unsigned *>(out)[0] = v;
    }
};

} // namespace causalflow::petit::rocm::quantization
