add_library(hal platform.cc)
set (HAL_DEPENDENCY absl::status)

if (WITH_ROCM)
  add_subdirectory(rocm)
  list(APPEND HAL_DEPENDENCY hal_rocm)
endif()

target_link_libraries(hal PUBLIC ${HAL_DEPENDENCY})