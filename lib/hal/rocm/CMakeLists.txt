add_library(hal_rocm OBJECT platform_rocm.cc)
target_link_libraries(hal_rocm absl::status hip::host)