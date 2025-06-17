#include "causalflow/petit/tal/algorithm.h"
#include "causalflow/petit/tal/tensor/layout.h"
#include "gemm/rocm/quantization/dequant.cuh"
#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"

#include <hip/hip_fp8.h>

namespace causalflow::petit::rocm::quantization::fp4 {

static constexpr unsigned kBits = 4;
static constexpr unsigned kPackFactor = 32 / kBits;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);

using _float8 = __hip_fp8_e4m3_fnuz;

struct RepackQWeightLayout128x16 {
    static constexpr unsigned kRowGroupSize = 16;
    static constexpr unsigned kLayoutM = 128;
    static constexpr unsigned kLayoutN = 16;
    static constexpr unsigned kGroupM = 128;
    static constexpr unsigned kGroupN = 64;

    __device__ auto GetQWeightShmLayout() const {
        using namespace causalflow::tal;
        using ShmShape =
            Shape<_4, Shape<C<kGroupM / kLayoutM>, C<kGroupN / kLayoutN>>,
                  Shape<_16, _4>>;
        using ShmStride = Stride<_1,
                                 Stride<C<kLayoutM / kPackFactor>,
                                        C<kGroupM * kLayoutN / kPackFactor>>,
                                 Stride<C<kGroupM / kPackFactor>, _4>>;
        using ShmLayout = Layout<ShmShape, ShmStride>;
        return ShmLayout{};
    }

    __device__ auto GetQWeightOutputLayout() const {
        using namespace causalflow::tal;

        using OutputShape =
            Shape<_1, _1, Shape<C<kGroupN / kLayoutN>, C<kGroupM / kLayoutM>>,
                  C<kWarpSize>>;
        auto output_stride = make_stride(
            out_chan_ * kGroupM / kPackFactor / kQuantVecSize,
            C<kLayoutM * kGroupN / kPackFactor / kQuantVecSize>{},
            make_stride(C<kWarpSize>{},
                        kLayoutM * out_chan_ / kPackFactor / kQuantVecSize),
            _1{});
        auto output_layout = make_layout(OutputShape{}, output_stride);
        return output_layout;
    }

    unsigned out_chan_;
};

__device__ static unsigned DequantShift(unsigned v) {
    unsigned r = 0;
    for (int i = 0; i < 8; i++) {
        unsigned shift = (3 - (i / 2)) + (i % 2) * 4;
        r |= ((v >> (i * 4)) & 0xf) << (shift * 4);
    }
    return r;
}

template <class Layout, unsigned kNumWarps, class ProcessWeightOp>
__global__ void RepackNvFp4ToPetitFp4WeightsKernel(
    ProcessWeightOp process, uint4 *__restrict__ output,
    const uint4 *__restrict__ input, unsigned in_chan, unsigned out_chan) {
    using namespace causalflow::tal;
    static constexpr unsigned kLayoutM = Layout::kLayoutM;
    static constexpr unsigned kLayoutN = Layout::kLayoutN;
    static constexpr unsigned kGroupM = Layout::kGroupM;
    static constexpr unsigned kGroupN = Layout::kGroupN;
    static constexpr unsigned kGroupInts = kGroupM * kGroupN / kPackFactor;
    static constexpr unsigned kThreads = kNumWarps * kWarpSize;

    Layout layout{out_chan};

    static_assert(kGroupM / kLayoutM * kGroupN / kLayoutN == kNumWarps, "");

    const unsigned tid = threadIdx.x, id_m = blockIdx.x, id_n = blockIdx.y;
    const unsigned wid = tid / kWarpSize, wtid = tid % kWarpSize;
    const uint4 *in_ptr =
        input + id_m * kGroupM / kPackFactor / kQuantVecSize +
        id_n * in_chan * kGroupN / kPackFactor / kQuantVecSize;

    __shared__ uint4 shm_qw[kGroupInts / kQuantVecSize];

    [[assume(tid < kThreads)]];
    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupInts / kQuantVecSize, kThreads) &&
         idx < kGroupInts / kQuantVecSize;
         i++, idx += kThreads) {
        unsigned row = idx / (kGroupM / kPackFactor / kQuantVecSize),
                 col = idx % (kGroupM / kPackFactor / kQuantVecSize);
        shm_qw[idx] = in_ptr[row * in_chan / kPackFactor / kQuantVecSize + col];
    }
    __syncthreads();

    auto shm_layout = layout.GetQWeightShmLayout();
    auto output_layout = layout.GetQWeightOutputLayout();

    const unsigned *sqw = reinterpret_cast<const unsigned *>(shm_qw);

    unsigned ret[4];
    for (int i = 0; i < 4; i++) {
        auto coord = shm_layout(make_coord(i, wid, wtid));
        unsigned qv = sqw[coord];
        ret[i] = process(qv);
    }

    auto output_coord = output_layout(make_coord(id_m, id_n, wid, wtid));
    output[output_coord] = *reinterpret_cast<const uint4 *>(ret);
}

template <class WeightLayout, unsigned kGroupM, unsigned kGroupN,
          unsigned kExpBias = 0>
__global__ void RepackNvFp4ScalesKernel(uint4 *__restrict__ out_scales,
                                        const uint4 *__restrict__ scales,
                                        unsigned in_chan, unsigned out_chan) {
    using namespace causalflow::tal;
    using Element = unsigned char;

    static constexpr unsigned kLayoutM = WeightLayout::kLayoutM;
    static constexpr unsigned kRowGroupSize = WeightLayout::kRowGroupSize;
    static constexpr unsigned kScaleVecSize = sizeof(uchar2) / sizeof(char);
    static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(Element);
    static constexpr unsigned kThreads = kWarpSize;

    const unsigned tid = threadIdx.x, id_m = blockIdx.x, id_n = blockIdx.y;

    const uint4 *in_scales =
        scales + id_m * kGroupM / kRowGroupSize / kVecSize +
        id_n * in_chan / kRowGroupSize * kGroupN / kVecSize;

    const half kMultiple = half(1 << kExpBias);

    static_assert(kGroupM / WeightLayout::kLayoutM == 2 &&
                      WeightLayout::kLayoutN == 16,
                  "");
    auto output_stride =
        make_stride(out_chan * kGroupM / kRowGroupSize / kScaleVecSize,
                    C<kLayoutM * kGroupN / kRowGroupSize / kScaleVecSize>{},
                    out_chan * kLayoutM / kRowGroupSize / kScaleVecSize, _1{});
    auto output_layout =
        make_layout(Shape<_1, _1, _2, C<kWarpSize>>{}, output_stride);

    __shared__ uint4 shm_scales[kGroupM / kRowGroupSize * kGroupN / kVecSize];

    for (unsigned i = 0, idx = tid;
         i < tal::CeilingDiv(kGroupM / kRowGroupSize * kGroupN / kVecSize,
                             kThreads) &&
         idx < kGroupM / kRowGroupSize * kGroupN / kVecSize;
         i++, idx += kThreads) {
        unsigned row = idx / (kGroupM / kRowGroupSize / kVecSize),
                 col = idx % (kGroupM / kRowGroupSize / kVecSize);
        shm_scales[idx] =
            in_scales[row * in_chan / kRowGroupSize / kVecSize + col];
    }

    __syncthreads();

    const unsigned short *shm =
        reinterpret_cast<const unsigned short *>(shm_scales);
    using ShmShape = Shape<Shape<_16, _4>, _2>;
    using ShmStride =
        Stride<Stride<C<kGroupM / kRowGroupSize / kScaleVecSize>, _1>, _4>;
    using ShmLayout = Layout<ShmShape, ShmStride>;

    auto shm_layout = ShmLayout{};
    for (int i = 0; i < 2; i++) {
        auto data = shm[shm_layout(make_coord(tid, i))];
        unsigned short ret;
        if constexpr (kExpBias != 0) {
            auto v = reinterpret_cast<const __hip_fp8_storage_t *>(&data);
            half2 h2;
            h2.x = __hip_cvt_fp8_to_halfraw(v[0], __HIP_E4M3);
            h2.y = __hip_cvt_fp8_to_halfraw(v[1], __HIP_E4M3);
            auto scaled = __hmul2(h2, half2{kMultiple, kMultiple});
            unsigned scaled_u32 = reinterpret_cast<const unsigned &>(scaled);
            // Convert the half2 scale to the E5M3 format.
            ret = ((scaled_u32 & 0xffff) >> 7) | ((scaled_u32 >> 23) << 8);
        } else {
            ret = data;
        }
        auto output_coord = output_layout(make_coord(id_m, id_n, i, tid));
        auto out_s2 = reinterpret_cast<unsigned short *>(out_scales);
        out_s2[output_coord] = ret;
    }
}

template <class Layout, class Dequantizer, class DequantizerForScale>
__global__ void DequantizePetitFp4Kernel(uint4 *__restrict__ output,
                                         const uint4 *__restrict__ input,
                                         const uint4 *__restrict__ scales,
                                         float global_scale, unsigned size_k,
                                         unsigned size_n) {
    using namespace causalflow::tal;
    using Element = typename Dequantizer::Element;
    using VectorType = typename Dequantizer::VectorType;
    using Content __attribute__((
        ext_vector_type(32 / (sizeof(uint) / sizeof(Element))))) = uint;
    static constexpr unsigned kLayoutM = Layout::kLayoutM;
    static constexpr unsigned kLayoutN = Layout::kLayoutN;
    static constexpr unsigned kRowGroupSize = Layout::kRowGroupSize;
    static constexpr unsigned kScaleVecSize = sizeof(uchar2) / sizeof(char);
    static constexpr unsigned kOutVecSize = sizeof(Content) / sizeof(Element);
    static constexpr unsigned kGroupM = kLayoutM;
    static constexpr unsigned kGroupN = kLayoutN;
    static constexpr bool kHighPrecision = !DequantizerForScale::kUpscale;

    const unsigned tid = threadIdx.x, id_k = blockIdx.x, id_n = blockIdx.y;

    auto stride_in = make_stride(
        size_n * kGroupM / kPackFactor / kQuantVecSize,
        C<kLayoutM * kGroupN / kPackFactor / kQuantVecSize>{}, _1{});
    auto layout_in = make_layout(Shape<_1, _1, C<kWarpSize>>{}, stride_in);

    auto stride_scale = make_stride(
        size_n * kGroupM / kRowGroupSize / kScaleVecSize,
        C<kGroupM * kGroupN / kRowGroupSize / kScaleVecSize>{}, _1{});
    auto layout_scale =
        make_layout(Shape<_1, _1, C<kWarpSize>>{}, stride_scale);

    auto stride_out =
        make_stride(C<kGroupM / kOutVecSize>{}, size_k * kGroupN / kOutVecSize,
                    make_stride(size_k / kOutVecSize, _1{}));
    auto layout_out = make_layout(Shape<_1, _1, Shape<_16, _4>>{}, stride_out);

    uint4 qw = input[layout_in(make_coord(id_k, id_n, tid))];
    unsigned short packed_scale = reinterpret_cast<const unsigned short *>(
        scales)[layout_scale(make_coord(id_k, id_n, tid))];

    const auto bias = Dequantizer::Bias(kHighPrecision);
    const VectorType bias2{bias, bias};

    VectorType ds;
    DequantizerForScale::DequantFullScale(&ds, packed_scale);
    const auto global_scale_f16 = Element(global_scale);
    VectorType gs2{global_scale_f16, global_scale_f16};

    VectorType ret[16];

    for (int i = 0; i < 4; i++) {
        const uint q = reinterpret_cast<const uint *>(&qw)[i];
        const Element s = i < 2 ? ds.x : ds.y;
        VectorType s2{s, s};
        VectorType dq[4];
        Dequantizer::Dequant(dq, q);
        Dequantizer::Dequant(dq + 2, q << 8);
        for (int j = 0; j < 4; j++) {
            if constexpr (kHighPrecision) {
                dq[j] = __hmul2(dq[j], bias2);
            }
            dq[j] = __hmul2(dq[j], s2);
            ret[i * 4 + j] = __hmul2(dq[j], gs2);
        }
    }

    reinterpret_cast<Content *>(
        output)[layout_out(make_coord(id_k, id_n, tid))] =
        *reinterpret_cast<Content *>(ret);
}

template <class Dequantizer, class DequantizerForScale>
__global__ void DequantizeNvFp4Kernel(uint4 *output, const uint4 *input,
                                      const uchar2 *scales, float global_scale,
                                      unsigned size_k, unsigned size_n) {
    using namespace causalflow::tal;
    using Element = typename Dequantizer::Element;
    using VectorType = typename Dequantizer::VectorType;
    static constexpr unsigned kGroupK = 128;
    static constexpr unsigned kGroupN = 16;
    static constexpr unsigned kRowGroupSize = 16;

    using Content __attribute__((
        ext_vector_type(32 / (sizeof(uint) / sizeof(Element))))) = uint;
    static constexpr unsigned kOutVecSize = sizeof(Content) / sizeof(Element);
    static constexpr unsigned kScaleVecSize = sizeof(uchar2) / sizeof(char);
    const unsigned tid = threadIdx.x, id_k = blockIdx.x, id_n = blockIdx.y;

    auto stride_in =
        make_stride(C<kGroupK / kPackFactor / kQuantVecSize>{},
                    size_k * kGroupN / kPackFactor / kQuantVecSize,
                    make_stride(_1{}, size_k / kPackFactor / kQuantVecSize));
    auto layout_in = make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_in);
    auto stride_out =
        make_stride(C<kGroupK / kOutVecSize>{}, size_k * kGroupN / kOutVecSize,
                    make_stride(_1{}, size_k / kOutVecSize));
    auto layout_out = make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_out);

    auto stride_scale =
        make_stride(C<kGroupK / kRowGroupSize / kScaleVecSize>{},
                    size_k / kRowGroupSize * kGroupN / kScaleVecSize,
                    make_stride(_1{}, size_k / kRowGroupSize / kScaleVecSize));
    auto layout_scale =
        make_layout(Shape<_1, _1, Shape<_4, _16>>{}, stride_scale);

    uint4 qw = input[layout_in(make_coord(id_k, id_n, tid))];
    unsigned short packed_scale = reinterpret_cast<const unsigned short *>(
        scales)[layout_scale(make_coord(id_k, id_n, tid))];

    const auto bias = Dequantizer::Bias(false);
    const VectorType bias2{bias, bias};

    // The channel scale is fp8e4m3 without offset
    static const unsigned short kScaleBiasU16 =
        (2 * (1 << (Dequantizer::kFp16Ex - 1)) - 8 - 1)
        << (16 - Dequantizer::kFp16Ex - 1);
    const Element scale_bias = reinterpret_cast<const Element &>(kScaleBiasU16);
    const VectorType scale_bias2{scale_bias, scale_bias};

    VectorType ds;
    DequantizerForScale::Dequant(&ds, packed_scale);
    ds = __hmul2(ds, scale_bias2);
    VectorType ret[16];

    const auto global_scale_f16 = Element(global_scale);
    VectorType gs2{global_scale_f16, global_scale_f16};

    for (int i = 0; i < 4; i++) {
        const uint q = reinterpret_cast<const uint *>(&qw)[i];
        unsigned q_shifted = DequantShift(q);
        VectorType dq[4];
        Dequantizer::Dequant(dq, q_shifted);
        Dequantizer::Dequant(dq + 2, q_shifted << 8);
        const Element s = i < 2 ? ds.x : ds.y;
        VectorType s2{s, s};
        for (int j = 0; j < 4; j++) {
            dq[j] = __hmul2(dq[j], bias2);
        }
        for (int j = 0; j < 4; j++) {
            ret[i * 4 + j] = __hmul2(dq[j], s2);
            ret[i * 4 + j] = __hmul2(ret[i * 4 + j], gs2);
        }
    }

    reinterpret_cast<Content *>(
        output)[layout_out(make_coord(id_k, id_n, tid))] =
        *reinterpret_cast<Content *>(ret);
}

int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n) {
    static constexpr unsigned kGroupM = 128;
    static constexpr unsigned kGroupN = 16;
    dim3 grid(k / kGroupM, n / kGroupN);
    dim3 block(kWarpSize);
    // We do not need to update the global scale as the kernel compute the bias
    // directly
    if (out_type == kDataTypeFp16) {
        using DQ = Dequantizer<half2, kDataTypeFp4e2m1>;
        using DS = DequantizerForFp8Scale<half2, false>;
        DequantizeNvFp4Kernel<DQ, DS><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uchar2 *>(scales), global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using DQ = Dequantizer<__hip_bfloat162, kDataTypeFp4e2m1>;
        using DS = DequantizerForFp8Scale<__hip_bfloat162, false>;
        DequantizeNvFp4Kernel<DQ, DS><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uchar2 *>(scales), global_scale, k, n);
    } else {
        return -1;
    }
    return 0;
}

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n) {
    using Layout = RepackQWeightLayout128x16;
    dim3 grid(k / Layout::kLayoutM, n / Layout::kLayoutN);
    dim3 block(kWarpSize);
    if (out_type == kDataTypeFp16) {
        using DQ = Dequantizer<half2, kDataTypeFp4e2m1>;
        using DS = DequantizerForFp8Scale<half2, true>;
        global_scale *= DS::GlobalScaleFactor();
        DequantizePetitFp4Kernel<Layout, DQ, DS><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uint4 *>(scales), global_scale, k, n);
    } else if (out_type == kDataTypeBf16) {
        using DQ = Dequantizer<__hip_bfloat162, kDataTypeFp4e2m1>;
        using DS = DequantizerForFp8Scale<__hip_bfloat162, true>;
        global_scale *= DS::GlobalScaleFactor();
        DequantizePetitFp4Kernel<Layout, DQ, DS><<<grid, block>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(input),
            reinterpret_cast<const uint4 *>(scales), global_scale, k, n);
    } else {
        return -1;
    }
    return 0;
}

void RepackNvFp4ToPetitFp4Weights(unsigned *output, const unsigned *input,
                                  unsigned in_chan, unsigned out_chan,
                                  hipStream_t stream) {
    using Layout = RepackQWeightLayout128x16;
    static constexpr unsigned kNumWarps = 4;
    dim3 grid(in_chan / Layout::kGroupM, out_chan / Layout::kGroupN);
    dim3 block(kNumWarps * kWarpSize);

    struct ProcessWeightOp {
        __device__ uint operator()(uint qv) const { return DequantShift(qv); }
    };
    ProcessWeightOp op;

    RepackNvFp4ToPetitFp4WeightsKernel<Layout, kNumWarps, ProcessWeightOp>
        <<<grid, block, 0, stream>>>(op, reinterpret_cast<uint4 *>(output),
                                     reinterpret_cast<const uint4 *>(input),
                                     in_chan, out_chan);
}

void RepackNvFp4ToPetitFp4Scales(unsigned *out_scales, const unsigned *scales,
                                 unsigned in_chan, unsigned out_chan,
                                 hipStream_t stream) {
    using Layout = RepackQWeightLayout128x16;
    static constexpr unsigned kGroupM = 2 * 128;
    static constexpr unsigned kGroupN = 16;
    dim3 scale_grid(in_chan / kGroupM, out_chan / kGroupN);
    dim3 block(kWarpSize);
    RepackNvFp4ScalesKernel<Layout, kGroupM, kGroupN, kFp8ScaleBias>
        <<<scale_grid, block, 0, stream>>>(
            reinterpret_cast<uint4 *>(out_scales),
            reinterpret_cast<const uint4 *>(scales), in_chan, out_chan);
}

} // namespace causalflow::petit::rocm::quantization::fp4