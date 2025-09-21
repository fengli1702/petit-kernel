#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"
#include "utils/hip_helper.h"
#include "utils/test_utils.h"
#include <fstream> 
#include <climits>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <bit> // For std::bit_cast
#include <algorithm> // For std::min
#include <sstream> // For robust printing

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

// Helper to print half/bfloat16 as a clean hex value
uint16_t to_u16(half val) { return std::bit_cast<uint16_t>(val); }
uint16_t to_u16(hip_bfloat16 val) { return std::bit_cast<uint16_t>(val); }

bool operator==(const hip_bfloat16 &a, const hip_bfloat16 &b) {
    return std::bit_cast<uint16_t>(a) == std::bit_cast<uint16_t>(b);
}

namespace causalflow::petit::rocm::quantization::fp4 {

// Forward declarations
int DequantNvFp4(unsigned *output, const unsigned *input,
                 const unsigned *scales, float global_scale, DataType out_type,
                 unsigned k, unsigned n);

int DequantPetitFp4(unsigned *output, const unsigned *input,
                    const unsigned *scales, float global_scale,
                    DataType out_type, unsigned k, unsigned n);

void RepackNvFp4ToPetitFp4Weights(unsigned *output, const unsigned *input,
                                  unsigned in_chan, unsigned out_chan,
                                  hipStream_t stream);

void RepackNvFp4ToPetitFp4Scales(unsigned *out_scales, const unsigned *scales,
                                 unsigned in_chan, unsigned out_chan,
                                 hipStream_t stream);

//static constexpr unsigned kVecSize = sizeof(uint2) / sizeof(char);
static constexpr unsigned kPackFactor = 32 / 4;
static constexpr unsigned kQuantVecSize = sizeof(uint4) / sizeof(unsigned);
static constexpr unsigned kRgsIn   = 32;  // mxFP4 输入的 groupsize
static constexpr unsigned kRgsOut  = 16;  // Petit / 解码端仍按 16
static constexpr unsigned kInVecBytes  = sizeof(uint2); // 8B = 8 个 8bit 缩放码
static constexpr unsigned kOutVecBytes = sizeof(uint4); // 16B = 16 个 8bit 缩放码

template <class Element, unsigned kM, unsigned kN> struct DeviceContext {
    using ScaleType = unsigned char;
    static constexpr unsigned kOutVecSize = sizeof(uint4) / sizeof(Element);
    uint4 d_weights_quant[kM * kN / kPackFactor / kQuantVecSize];
    uint2 d_scales[kM * kN / kRgsIn  / (kInVecBytes  / sizeof(char))];
    uint4 d_reference[kM * kN / kOutVecSize];
    uint4 d_petit_weights[kM * kN / kPackFactor / kQuantVecSize];
    uint4 d_petit_scales[kM * kN / kRgsOut / (kOutVecBytes / sizeof(char))];
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
    std::vector<ScaleType> h_scales(kM * kN / kRgsIn);
    
    std::mt19937 gen(42);
    std::uniform_int_distribution<unsigned> dist_q(0, UINT_MAX);
    auto gen_q = [&]() { return dist_q(gen); };

    std::uniform_int_distribution<unsigned> dist_scale(118 , 130);
    auto gen_scale_fp8 = [&]() { return dist_scale(gen); };

    FillRandomValue(gen_q, &h_qweights);
    FillRandomValue(gen_scale_fp8, &h_scales);

    //// ====================== <<< PRINTING LOGIC (FULLY FIXED & ROBUST) >>> ======================
    //std::cout << "\n--- [INPUT DATA INSPECTION] ---" << std::endl;
    //
    //std::stringstream ss;
////
    //const size_t num_weights_to_print = 256;
    //std::cout << "First " << std::dec << num_weights_to_print << " Packed Weight uints (h_qweights):" << std::endl;
    //for (size_t i = 0; i < std::min(num_weights_to_print, h_qweights.size()); ++i) {
    //    ss.str("");
    //    ss << "  [" << std::dec << i << "]: 0x" << std::hex << std::setw(8) << std::setfill('0') << h_qweights[i];
    //    std::cout << ss.str() << std::endl;
    //}
//
    //const size_t num_scales_to_print = 128;
    //std::cout << "\nFirst " << std::dec << num_scales_to_print << " Packed Scales (h_scales):" << std::endl;
    //for (size_t i = 0; i < std::min(num_scales_to_print, h_scales.size()); ++i) {
    //    ss.str("");
    //    ss << "  [" << std::dec << i << "]: 0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(h_scales[i]);
    //    std::cout << ss.str() << std::endl;
    //}
    //std::cout << "--------------------------------\n" << std::endl;
    //// =======================================================================================
////
    //// ... (rest of the function is the same)
    //std::cout << "[GT Generation] Saving raw input data to output.txt..." << std::endl;
    //
    //// Save all input data to output.txt in the original format
    //std::ofstream output_file("output.txt");
    //if (output_file.is_open()) {
    //    // Save weights
    //    output_file << "First " << h_qweights.size() << " Packed Weight uints (h_qweights):" << std::endl;
    //    for (size_t i = 0; i < h_qweights.size(); ++i) {
    //        output_file << "  [" << std::dec << i << "]: 0x" 
    //                   << std::hex << std::setw(8) << std::setfill('0') << h_qweights[i] << std::endl;
    //    }
    //    
    //    // Save scales
    //    output_file << "\nFirst " << h_scales.size() << " Packed Scales (h_scales):" << std::endl;
    //    for (size_t i = 0; i < h_scales.size(); ++i) {
    //        output_file << "  [" << std::dec << i << "]: 0x" 
    //                   << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(h_scales[i]) << std::endl;
    //    }
    //    
    //    output_file << "\n---" << std::endl;
    //    output_file.close();
    //    std::cout << "   Input data saved to output.txt" << std::endl;
    //} else {
    //    std::cout << "   Failed to open output.txt for input data" << std::endl;
    //}
    //
    //std::cout << "[GT Generation] ...Done." << std::endl;

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

    //// ====================== <<< APPEND OUTPUT DATA TO FILE >>> ======================
    //std::cout << "\n[GT Generation] Appending output data to output.txt..." << std::endl;
    //
    //// Append output data to the existing output.txt file
    //std::ofstream output_file("output.txt", std::ios::app);
    //if (output_file.is_open()) {
    //    // Save reference outputs - ALL outputs from 0 to end
    //    output_file << "\nFirst " << h_reference.size() << " outputs from NV Path (Reference):" << std::endl;
    //    for (size_t i = 0; i < h_reference.size(); ++i) {
    //        output_file << "Ref   [" << std::setw(3) << std::dec << i << "]: 0x" 
    //                   << std::hex << std::setw(4) << std::setfill('0') << to_u16(h_reference[i]) << std::endl;
    //    }
    //    
    //    // Save petit outputs - ALL outputs from 0 to end
    //    output_file << "\nFirst " << h_petit_output.size() << " outputs from Petit Path:" << std::endl;
    //    for (size_t i = 0; i < h_petit_output.size(); ++i) {
    //        output_file << "Petit [" << std::setw(3) << std::dec << i << "]: 0x" 
    //                   << std::hex << std::setw(4) << std::setfill('0') << to_u16(h_petit_output[i]) << std::endl;
    //    }
    //    
    //    output_file << "\n---" << std::endl;
    //    output_file.close();
    //    std::cout << "   All " << h_reference.size() << " output data appended to output.txt" << std::endl;
    //} else {
    //    std::cout << "   Failed to open output.txt for appending output data" << std::endl;
    //}
    
    //std::cout << "[GT Generation] Output data saved successfully." << std::endl;
    //// ================================================================
//
    //// 
    //std::cout << "\n--- [COMBINED PRINT AND VERIFY] ---" << std::endl;
    
    for (size_t i = 0; i < kM * kN; ++i) {
        
        EXPECT_EQ(h_reference[i], h_petit_output[i])
                << "Mismatch at index " << i;
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

TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Bf16) {
    TestConvert<hip_bfloat16, 512, 512>(1.0, kDataTypeBf16);
}

TEST_F(NvFp4ToPetitFp4Test, TestLayout128x16Fp16) {
    TestConvert<half, 512, 512>(1.0, kDataTypeFp16);
}
} // namespace