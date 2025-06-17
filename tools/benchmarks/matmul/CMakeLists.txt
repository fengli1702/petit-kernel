include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_executable(bench_matmul main.cc matmul.cc)

unset(BENCH_MATMUL_DEPS_PRIVATE)

if (WITH_ROCM)
  add_subdirectory(rocm)
  list(APPEND BENCH_MATMUL_DEPS_PRIVATE benchmark_matmul_rocm)
endif ()

target_link_libraries(bench_matmul PUBLIC absl::status hal gflags fmt::fmt PRIVATE ${BENCH_MATMUL_DEPS_PRIVATE})