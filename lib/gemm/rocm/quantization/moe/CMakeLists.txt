# lib/gemm/rocm/quantization/moe/CMakeLists.txt

# MoE模块的GPU kernel源文件
set(MOE_GPU_KERNEL_SRCS
    moe_gemm_fp4_grid.cc
)

# 设置GPU kernel源文件属性
set_source_files_properties(${MOE_GPU_KERNEL_SRCS} PROPERTIES
    LANGUAGE HIP
)

# 创建MoE量化库
add_library(causalflow-petit-rocm-moe-quantization
    ${MOE_GPU_KERNEL_SRCS}
)

# 设置包含目录
target_include_directories(causalflow-petit-rocm-moe-quantization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}         # 包含当前目录（moe/）
    ${CMAKE_CURRENT_SOURCE_DIR}/..      # 包含上级目录（quantization/）
    ${CMAKE_CURRENT_SOURCE_DIR}/../..   # 包含gemm/rocm/
)

# 链接依赖
target_link_libraries(causalflow-petit-rocm-moe-quantization 
    PRIVATE
        hip::host
        causalflow-petit-tests  # 用于测试工具
)

set_source_files_properties(moe_gemm_fp4_test.cc PROPERTIES
    LANGUAGE HIP
)
# 添加测试可执行文件
add_executable(moe_fp4_fp16_rocm_test moe_gemm_fp4_test.cc)
target_link_libraries(moe_fp4_fp16_rocm_test 
    PRIVATE
        causalflow-petit-rocm-moe-quantization
        causalflow-petit-tests
        roc::hipblaslt
        GTest::gmock 
        GTest::gtest_main
)
add_test(NAME moe_fp4_fp16_rocm_test COMMAND moe_fp4_fp16_rocm_test)