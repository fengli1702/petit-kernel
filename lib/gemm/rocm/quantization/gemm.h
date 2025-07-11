#pragma once

#include "types.h"
#include <hip/hip_runtime.h>

namespace causalflow::petit::rocm::quantization {

enum MatmulFeatures {
    kMatmulFeatures_Global = 0,
    kMatmulFeatures_Grid = 1,
    kMatmulFeatures_HighPrecision = 1 << 1,
};

enum MatmulPipeline {
    kMatmulPipeline_1,
    kMatmulPipeline_2,
};

enum MatmulElementB {
    kMatmulTypeBInt4,
    kMatmulTypeBFp4,
};

enum MatmulMfmaType {
    kMatmulMfmaTypeFp16,
    kMatmulMfmaTypeBf16,
    kMatmulMfmaTypeFp8,
};

enum MatmulWarpPartition {
    // Each warp handles the full tile
    kMatmulWarpPartition_NK,
    // Warps handle the tile cooperatively
    kMatmulWarpPartition_Cooperative,
};

struct SolutionId {
    unsigned tile_m : 8;
    unsigned tile_n : 8;
    // number of K tiles / 4 as the unit is 64
    unsigned tile_k : 8;
    MatmulFeatures features : 4;
    MatmulPipeline pipeline : 4;
    MatmulElementB element_b : 4;
    MatmulMfmaType mfma_type : 4;
    unsigned warp_partition_m : 4;
    unsigned warp_partition_n : 4;
    unsigned warp_partition_k : 4;
    MatmulWarpPartition warp_partition : 4;
    unsigned padding : 8;

    constexpr unsigned long Repr() const {
        return *(const unsigned long *)(this);
    }

    static constexpr SolutionId FromRepr(unsigned long repr) {
        return SolutionId{
            .tile_m = static_cast<unsigned int>((repr >> 0) & 0xff),
            .tile_n = static_cast<unsigned int>((repr >> 8) & 0xff),
            .tile_k = static_cast<unsigned int>((repr >> 16) & 0xff),
            .features = static_cast<MatmulFeatures>((repr >> 24) & 0xf),
            .pipeline = static_cast<MatmulPipeline>((repr >> 28) & 0xf),
            .element_b = static_cast<MatmulElementB>((repr >> 32) & 0xf),
            .mfma_type = static_cast<MatmulMfmaType>((repr >> 36) & 0xf),
            .warp_partition_m = static_cast<unsigned int>((repr >> 40) & 0xf),
            .warp_partition_n = static_cast<unsigned int>((repr >> 44) & 0xf),
            .warp_partition_k = static_cast<unsigned int>((repr >> 48) & 0xf),
            .warp_partition =
                static_cast<MatmulWarpPartition>((repr >> 52) & 0xf),
            .padding = 0,
        };
    }

    static constexpr SolutionId
    MultiStage(MatmulPipeline pipeline, MatmulFeatures features,
               MatmulElementB element_b, MatmulMfmaType mfma_type,
               unsigned tile_m, unsigned tile_n, unsigned tile_k,
               MatmulWarpPartition warp_partition, unsigned warp_partition_m,
               unsigned warp_partition_n, unsigned warp_partition_k) {
        return SolutionId{tile_m,           tile_n,           tile_k / 4,
                          features,         pipeline,         element_b,
                          mfma_type,        warp_partition_m, warp_partition_n,
                          warp_partition_k, warp_partition,   0};
    }

    static constexpr SolutionId Default() {
        return SolutionId{
            1,
            4,
            2,
            kMatmulFeatures_Grid,
            kMatmulPipeline_2,
            kMatmulTypeBFp4,
            kMatmulMfmaTypeFp16,
            1,
            2,
            2,
            kMatmulWarpPartition_NK,
        };
    }
};
static_assert(sizeof(SolutionId) == 8, "");

static constexpr int kErrorProblemShape = 1;
static constexpr int kErrorKernelShape = 2;

//
// Describe hints of selecting algorithms, ignored when solution_id is set
struct PetitSolutionHints {
    DataType a_type;
    DataType b_type;
    DataType c_type;
    bool require_high_precision;
};

namespace fp4 {
int GemmFp4Fp16Grid(unsigned *c, const unsigned *a, const unsigned *b,
                    const unsigned *scales, const float *global_scale,
                    const unsigned m, const unsigned n, const unsigned k,
                    const PetitSolutionHints &hints, unsigned long solution_id,
                    hipStream_t stream);

int GemmGetSolutions(const PetitSolutionHints &hints, unsigned m, unsigned n,
                     unsigned k, SolutionId *sols, unsigned *n_sols);

void RepackNvFp4ToPetitFp4Weights(unsigned *output, const unsigned *input,
                                  unsigned in_chan, unsigned out_chan,
                                  hipStream_t stream);

void RepackNvFp4ToPetitFp4Scales(unsigned *out_scales, const unsigned *scales,
                                 unsigned in_chan, unsigned out_chan,
                                 hipStream_t stream);
} // namespace fp4

} // namespace causalflow::petit::rocm::quantization
