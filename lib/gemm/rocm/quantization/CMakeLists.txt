set(GPU_KERNEL_SRCS
  fp4/quantization_utils.cu
  fp4/gemm_fp4_fp16_grid.cu
)

set_source_files_properties(${GPU_KERNEL_SRCS} PROPERTIES
  LANGUAGE HIP
)

add_library(causalflow-petit-rocm-quantization
  fp4/algo_chooser.cc
  ${GPU_KERNEL_SRCS}
)
target_link_libraries(causalflow-petit-rocm-quantization hip::host)

add_subdirectory(fp4)