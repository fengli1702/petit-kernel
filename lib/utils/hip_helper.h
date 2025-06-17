#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>

namespace causalflow {

static inline void CheckHIPStatus(hipError_t status) {
    if (status != hipSuccess) {
        std::cerr << "HIP Error: " << hipGetErrorString(status) << std::endl;
        throw std::runtime_error("CUDA Error");
    }
}

} // namespace causalflow