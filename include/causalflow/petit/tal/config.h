#pragma once

#include "detail/helper_macro.h"

namespace causalflow::tal {

[[noreturn]] inline TAL_HOST_DEVICE void unreachable() {
    // Uses compiler specific extensions if possible.
    // Even if no extension is used, undefined behavior is still raised by
    // an empty function body and the noreturn attribute.
#if defined(_MSC_VER) && !defined(__clang__) // MSVC
    __assume(false);
#else // GCC, Clang
    __builtin_unreachable();
#endif
}

#if defined(__CUDA_ARCH__) || defined(_NVHPC_CUDA)
#define TAL_INLINE_CONSTANT static const __device__
#else
#define TAL_INLINE_CONSTANT static constexpr
#endif

} // namespace causalflow::tal