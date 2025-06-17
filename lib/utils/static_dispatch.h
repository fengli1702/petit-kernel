#pragma once

#define STATIC_BOOL(cond, const_name, ...)                                     \
    [&] {                                                                      \
        if (cond) {                                                            \
            constexpr static bool const_name = true;                           \
            return __VA_ARGS__();                                              \
        } else {                                                               \
            constexpr static bool const_name = false;                          \
            return __VA_ARGS__();                                              \
        }                                                                      \
    }()
