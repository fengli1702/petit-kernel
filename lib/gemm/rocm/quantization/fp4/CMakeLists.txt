add_executable(quantization_utils_fp4_rocm_test quantization_utils_fp4_test.cc)
target_link_libraries(quantization_utils_fp4_rocm_test causalflow-petit-rocm-quantization GTest::gtest_main)
add_test(quantization_utils_fp4_rocm_test quantization_utils_fp4_rocm_test)

add_executable(gemm_fp4_fp16_rocm_test gemm_fp4_fp16_rocm_test.cc)
target_link_libraries(gemm_fp4_fp16_rocm_test causalflow-petit-rocm-quantization causalflow-petit-tests roc::hipblaslt GTest::gtest_main)
add_test(gemm_fp4_fp16_rocm_test gemm_fp4_fp16_rocm_test)

add_executable(gemm_fp4_fp16_gen_solution_list gen_solution_list.cc)
target_link_libraries(gemm_fp4_fp16_gen_solution_list hip::host gflags fmt::fmt)

set (CODEGEN_EXECUTABLE $<TARGET_FILE:gemm_fp4_fp16_gen_solution_list>)
set (CODEGEN_CMAKE_FRAGMENTS ${CMAKE_CURRENT_BINARY_DIR}/kernel_source.cmake)

add_custom_command(
    OUTPUT ${CODEGEN_CMAKE_FRAGMENTS}
    COMMAND ${CODEGEN_EXECUTABLE} --source_list_cmake ${CODEGEN_CMAKE_FRAGMENTS} --source_list solutions.inl --output_dir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS gemm_fp4_fp16_gen_solution_list
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating kernels"
    VERBATIM
)

add_custom_target(gen_fp4_fp16_kernel DEPENDS ${CODEGEN_CMAKE_FRAGMENTS})

if(NOT EXISTS ${CODEGEN_CMAKE_FRAGMENTS})
    file (WRITE ${CODEGEN_CMAKE_FRAGMENTS} "# Bootstrap fragments - Will be regenerated")
endif()

include(${CODEGEN_CMAKE_FRAGMENTS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(GPU_KERNEL_SRCS
  quantization_utils.cu
  gemm_fp4_fp16_grid.cc
)

set_source_files_properties(${GPU_KERNEL_SRCS} PROPERTIES
    LANGUAGE HIP
)

if (DEFINED GEMM_FP4_FP16_SRCS)
    list(TRANSFORM GEMM_FP4_FP16_SRCS
        PREPEND ${CMAKE_CURRENT_BINARY_DIR}/
        OUTPUT_VARIABLE GEMM_FP4_FP16_GEN_SRCS
    )
    set_source_files_properties(${GEMM_FP4_FP16_GEN_SRCS} PROPERTIES
        LANGUAGE HIP
        COMPILE_FLAGS "-include ${CMAKE_CURRENT_SOURCE_DIR}/solution_impl.h" 
        GENERATED TRUE
    )
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/solutions.inl PROPERTIES
        GENERATED TRUE
    )
endif()

add_library(causalflow-petit-rocm-quantization
  algo_chooser.cc
  solution_map.cc
  ${GPU_KERNEL_SRCS}
  ${GEMM_FP4_FP16_GEN_SRCS}
)

target_link_libraries(causalflow-petit-rocm-quantization hip::host)
add_dependencies(causalflow-petit-rocm-quantization gen_fp4_fp16_kernel)
