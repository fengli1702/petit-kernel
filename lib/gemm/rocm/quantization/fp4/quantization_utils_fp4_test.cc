#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"

#include <climits>
#include <fstream>
#include <gtest/gtest.h>
#include <random>

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

bool operator==(const hip_bfloat16 &a, const hip_bfloat16 &b) {
    return __builtin_bit_cast(uint16_t, a) == __builtin_bit_cast(uint16_t, b);
}

namespace causalflow::petit::rocm::quantization::fp4 {

int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n);

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n);

static constexpr unsigned kVecSize = sizeof(uint4) / sizeof(char);
static constexpr unsigned kPackFactor = 32 / 4;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);
static constexpr unsigned kRowGroupSize = 32;

template <class Element, unsigned kM, unsigned kN> struct DeviceContext {
    using ScaleType = unsigned char;
    static constexpr unsigned kOutVecSize = sizeof(uint4) / sizeof(Element);
    uint4 d_weights_quant[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_reference[kM * kN / kOutVecSize];
    uint4 d_petit_weights[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_petit_scales[kM * kN / kRowGroupSize / kVecSize];
    uint4 d_output[kM * kN / kOutVecSize];

    static DeviceContext *PrepareDevice();
    void CompareOutputsFromDevice() const;
};

template <class Element, unsigned kM, unsigned kN>
DeviceContext<Element, kM, kN> *
DeviceContext<Element, kM, kN>::PrepareDevice() {
    DeviceContext<Element, kM, kN> *d_ctx;
    CheckHIPStatus(hipMalloc(&d_ctx, sizeof(DeviceContext<Element, kM, kN>)));

    std::vector<unsigned> h_qweights(kM * kN / kPackFactor);
    std::vector<ScaleType> h_scales(kM * kN / kRowGroupSize);

    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
    auto gen_q = [&]() { return dist_q(gen); };

    // Only generate positive scales based on how preprocessing of the scales is
    // done
    std::uniform_int_distribution<unsigned> dist_scale(1, 126);
    auto gen_scale_fp8 = [&]() { return dist_scale(gen); };

    FillRandomValue(gen_q, &h_qweights);
    FillRandomValue(gen_scale_fp8, &h_scales);

    CheckHIPStatus(hipMemcpy(d_ctx->d_weights_quant, h_qweights.data(),
                             h_qweights.size() * sizeof(unsigned),
                             hipMemcpyHostToDevice));
    CheckHIPStatus(hipMemcpy(d_ctx->d_scales, h_scales.data(),
                             h_scales.size() * sizeof(ScaleType),
                             hipMemcpyHostToDevice));
    return d_ctx;
}

template <class Element, unsigned kM, unsigned kN>
void DeviceContext<Element, kM, kN>::CompareOutputsFromDevice() const {
    std::vector<Element> h_reference(kM * kN), h_petit_output(kM * kN);
    CheckHIPStatus(hipMemcpy(h_reference.data(), d_reference,
                             h_reference.size() * sizeof(Element),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipMemcpy(h_petit_output.data(), d_output,
                             h_petit_output.size() * sizeof(Element),
                             hipMemcpyDeviceToHost));
    CheckHIPStatus(hipDeviceSynchronize());

    for (unsigned i = 0; i < 96; ++i) {
        EXPECT_EQ(h_reference[i], h_petit_output[i])
            << "Output and reference differ at index " << i;
    }
}

class NvFp4ToPetitFp4Test : public ::testing::Test {
  public:
    template <class Element, unsigned kM, unsigned kN>
    void TestConvert(float global_scale, DataType out_type) const {
        auto d_ctx = DeviceContext<Element, kM, kN>::PrepareDevice();

        DequantNvFp4(reinterpret_cast<unsigned *>(d_ctx->d_reference),
                     reinterpret_cast<const unsigned *>(d_ctx->d_weights_quant),
                     reinterpret_cast<const unsigned *>(d_ctx->d_scales),
                     global_scale, out_type, kM, kN);

        RepackNvFp4ToPetitFp4Weights(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_weights_quant), kM, kN,
            nullptr);

        RepackNvFp4ToPetitFp4Scales(
            reinterpret_cast<unsigned *>(d_ctx->d_petit_scales),
            reinterpret_cast<const unsigned *>(d_ctx->d_scales), kM, kN,
            nullptr);

        DequantPetitFp4(
            reinterpret_cast<unsigned *>(d_ctx->d_output),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_weights),
            reinterpret_cast<const unsigned *>(d_ctx->d_petit_scales),
            global_scale, out_type, kM, kN);

        d_ctx->CompareOutputsFromDevice();
        CheckHIPStatus(hipFree(d_ctx));
    }
};

//TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Bf16) {
//    TestConvert<hip_bfloat16, 512, 512>(1.0, kDataTypeBf16);
//}

TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Fp16) {
    TestConvert<half, 512, 512>(1.0, kDataTypeFp16);
}

} // namespace causalflow::petit::rocm::quantization::fp4