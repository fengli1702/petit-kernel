#pragma once

#include "gemm/rocm/amd_fastmath.cuh"
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

__device__ static inline void Fp4ToFp16(unsigned o[4], unsigned q) {
    unsigned qr = __builtin_bitreverse32(q);
    o[0] = q & 0x8e008e00;
    o[1] = (q << 8) & 0x8e008e00;
    o[2] = qr & 0x8e008e00;
    o[3] = (qr << 8) & 0x8e008e00;
}

__device__ static inline void Fp4ToBf8(unsigned o[2], unsigned q) {
    unsigned qr = __builtin_bitreverse32(q);
    o[0] = q & 0x8e8e8e8e;
    o[1] = qr & 0x8e8e8e8e;
}

__device__ inline half e8m0_to_half(unsigned char e8m0_val) {
    uint16_t fp16_bits;
    if (e8m0_val == 0xFFu) {
        fp16_bits = 0x7C01u;
    } else {
        const int bias_e8m0 = 127;
        const int bias_fp16 = 15;
        int exp_unbiased = static_cast<int>(e8m0_val) - bias_e8m0;
        int exp_fp16 = exp_unbiased + bias_fp16;
        if (exp_fp16 <= 0) {
            fp16_bits = 0x0000u;
        } else if (exp_fp16 >= 0x1F) {
            fp16_bits = 0x7C00u;
        } else {
            fp16_bits = static_cast<uint16_t>( (exp_fp16 & 0x1F) << 10 );
        }
    }
    return __ushort_as_half(fp16_bits);
}

__device__ inline __hip_bfloat16 e8m0_to_bfloat16(unsigned char e8m0_val) {
    uint16_t bf16_bits;
    if (e8m0_val == 0xFFu) {
        bf16_bits = static_cast<uint16_t>((0xFFu << 7) | 0x1u);
    } else {
        bf16_bits = static_cast<uint16_t>(static_cast<uint16_t>(e8m0_val) << 7);
    }
    return reinterpret_cast<__hip_bfloat16&>(bf16_bits);
}

} // namespace detail

template <>
struct Dequantizer<half2, kDataTypeFp4e2m1>
    : public detail::DequantizerToFp16Impl<half2, 2, 5> {
    using Element = half;
    using VectorType = half2;

    __device__ static Element Bias(bool high_precision) {
        //const unsigned off =
        //    high_precision ? kExpOffset - kFp8ScaleBias : kExpOffset;
        const unsigned off = kExpOffset;
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

template <bool kHighPrecision> struct UnifiedDequantizerForFp4Fp16 {
    using UnpackedScale = half;
    using Element = half;
    using UnpackedData = half2[4];

    using DQ = Dequantizer<half2, kDataTypeFp4e2m1>;
    //using DS = DequantizerForFp8Scale<half2, !kHighPrecision>;

    __device__ static UnpackedScale DequantScales(unsigned char s) {
        //UnpackedScale ds;
        //DS::DequantFullScale(&ds, s);
        //return ds;
        return detail::e8m0_to_half(s);
    }

    __device__ static void DequantWithScale(UnpackedData &out, unsigned q,
                                            Element scale) {
        half2 s2{scale, scale};
        const auto bias = DQ::Bias(kHighPrecision);
        const half2 bias2{bias, bias};
        detail::Fp4ToFp16(reinterpret_cast<unsigned *>(&out), q);

        for (int i = 0; i < 4; i++) {
            if constexpr (kHighPrecision) {
                out[i] = fastmath::hmul2(out[i], bias2);
            }
            out[i] = fastmath::hmul2(out[i], s2);
        }
    }
};

template <bool kHighPrecision> struct UnifiedDequantizerForFp4Bf16 {
    using UnpackedScale = __hip_bfloat16;
    using Element = __hip_bfloat16;
    using UnpackedData = __hip_bfloat162[4];

    using DQ = Dequantizer<__hip_bfloat162, kDataTypeFp4e2m1>;
    //using DS = DequantizerForFp8Scale<__hip_bfloat162, !kHighPrecision>;

    __device__ static UnpackedScale DequantScales(unsigned char s) {
        //using FP8Scale = DequantizerForFp8Scale<half2, false>;
        //half2 ds;
        //FP8Scale::Dequant(&ds, s);
        //return ds;
        return detail::e8m0_to_bfloat16(s);
    }

    __device__ static void DequantWithScale(UnpackedData &out, unsigned q,
                                            Element scale) {// Element is __hip_bfloat16
#if defined(__gfx942__) || defined(__gfx950__)
        DequantWithScaleImplBf8(out, q, scale);
#else
        DequantWithScaleImplFp16(out, q, scale);
#endif
    }

  private:
    __device__ static void DequantWithScaleImplFp16(UnpackedData &out,
                                                    unsigned q, Element scale);
    __device__ static void DequantWithScaleImplBf8(UnpackedData &out,
                                                   unsigned q, Element scale);
};

template <bool kHighPrecision>
__device__ void
UnifiedDequantizerForFp4Bf16<kHighPrecision>::DequantWithScaleImplFp16(
    UnpackedData &out, unsigned q, Element scale) {

    const float2 s2{__bfloat162float(scale), __bfloat162float(scale)};

    // Since internallly it is FP16, the bias is the same as the FP16 bias
    // For high precision we divide by 2 ** 7 to undo the preprocessing of
    // the scales
    //static constexpr unsigned kBias = kHighPrecision ? 0x43000000  // 2** 7
    //                                                 : 0x46800000; // 2 ** 14
    //const float2 bias_f32_2{reinterpret_cast<const float &>(kBias),
    //                        reinterpret_cast<const float &>(kBias)};
    //
    const auto bias_bf16 = DQ::Bias(kHighPrecision); 
    const float bias_f32 = __bfloat162float(bias_bf16);
    const float2 bias_f32_2{bias_f32, bias_f32};
    detail::Fp4ToFp16(reinterpret_cast<unsigned *>(&out), q);
    const half2 *h2 = reinterpret_cast<const half2 *>(&out);
    float2 out_f2[4];
    for (int i = 0; i < 4; i++) {
        out_f2[i].x = __half2float(h2[i].x);
        out_f2[i].y = __half2float(h2[i].y);
    }

    for (int i = 0; i < 4; i++) {
        //if constexpr (kHighPrecision) {
        out_f2[i] = amdgcn_pk_mul_f32(out_f2[i], bias_f32_2);
        //}
        out_f2[i] = amdgcn_pk_mul_f32(out_f2[i], s2);
    }

    unsigned *o = reinterpret_cast<unsigned *>(&out);
    for (int i = 0; i < 4; i++) {
        const uint2 *f2 = reinterpret_cast<const uint2 *>(&out_f2[i]);
        o[i] = amdgcn_perm_b32(f2[0].y, f2[0].x, 0x07060302);
    }
}

template <bool kHighPrecision>
__device__ void
UnifiedDequantizerForFp4Bf16<kHighPrecision>::DequantWithScaleImplBf8(
    UnpackedData &out, unsigned q, Element scale) {
#if defined(__gfx942__) || defined(__gfx950__)
    const float2 s2{__bfloat162float(scale), __bfloat162float(scale)};

    // Since internallly it is FP16, the bias is the same as the FP16 bias
    // For high precision we divide by 2 ** 7 to undo the preprocessing of
    // the scales
    //static constexpr unsigned kBias = kHighPrecision ? 0x43000000  // 2** 7
    //                                                 : 0x46800000; // 2 ** 14
    //const float2 bias_f32_2{reinterpret_cast<const float &>(kBias),
    //                        reinterpret_cast<const float &>(kBias)};
//
    const auto bias_bf16 = DQ::Bias(kHighPrecision);
    const float bias_f32 = __bfloat162float(bias_bf16);
    const float2 bias_f32_2{bias_f32, bias_f32};
    unsigned bf8[2];
    detail::Fp4ToBf8(bf8, q);

    float2 out_f2[4];
    auto *out_v2f = reinterpret_cast<v2f *>(&out);
    for (int i = 0; i < 2; i++) {
        out_v2f[i * 2] = __builtin_amdgcn_cvt_pk_f32_bf8(bf8[i], false);
        out_v2f[i * 2 + 1] = __builtin_amdgcn_cvt_pk_f32_bf8(bf8[i], true);
    }

    for (int i = 0; i < 4; i++) {
        //if constexpr (kHighPrecision) {
        out_f2[i] = amdgcn_pk_mul_f32(out_f2[i], bias_f32_2);
        //}
        out_f2[i] = amdgcn_pk_mul_f32(out_f2[i], s2);
    }

    unsigned *o = reinterpret_cast<unsigned *>(&out);
    o[0] = amdgcn_perm_b32(out_f2[1].y, out_f2[0].y, 0x07060302);
    o[1] = amdgcn_perm_b32(out_f2[1].x, out_f2[0].x, 0x07060302);
    o[2] = amdgcn_perm_b32(out_f2[3].x, out_f2[2].x, 0x07060302);
    o[3] = amdgcn_perm_b32(out_f2[3].y, out_f2[2].y, 0x07060302);
#endif
}

} // namespace causalflow::petit::rocm::quantization