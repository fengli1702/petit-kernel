#pragma once

#if defined(__CUDACC__) || defined(__HIPCC__)
#define TAL_HOST_DEVICE __host__ __device__
#define TAL_DEVICE __device__
#else
#define TAL_HOST_DEVICE
#define TAL_DEVICE
#endif