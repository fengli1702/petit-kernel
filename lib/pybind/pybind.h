#pragma once

#include <torch/all.h>
#include <torch/python.h>

namespace causalflow::petit::rocm::quantization {
struct PetitSolutionHints;
}

namespace causalflow::petit::pybind {

torch::Tensor RepackNvFp4(torch::Tensor &b_q_weight, int64_t size_n,
                          int64_t size_k);

torch::Tensor ProcessNvFp4Scales(torch::Tensor &scales, int64_t size_n,
                                 int64_t size_k);

torch::Tensor MulFp4A16(const torch::Tensor &A, const torch::Tensor &B,
                        const torch::Tensor &s, float global_scale,
                        int64_t size_m, int64_t size_n, int64_t size_k,
                        int64_t solution_id);

py::list GetFp4Solutions(
    const causalflow::petit::rocm::quantization::PetitSolutionHints &hints,
    int64_t size_m, int64_t size_n, int64_t size_k);

} // namespace causalflow::petit::pybind