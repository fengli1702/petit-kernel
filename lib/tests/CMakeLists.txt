add_library(causalflow-petit-tests quantization.cc)
target_link_libraries(causalflow-petit-tests hal absl::status)