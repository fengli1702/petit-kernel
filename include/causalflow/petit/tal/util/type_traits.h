#pragma once

#include <cstddef>
#include <utility>

namespace causalflow::tal {

#define __TAL_REQUIRES(...)                                                    \
    typename std::enable_if<(__VA_ARGS__)>::type * = nullptr
#define __TAL_REQUIRES_V(...)                                                  \
    typename std::enable_if<decltype((__VA_ARGS__))::value>::type * = nullptr

using std::declval;
using std::remove_const_t;
using std::remove_cv_t;
using std::remove_cvref;
using std::remove_cvref_t;
using std::remove_reference_t;
using std::void_t;

using std::is_arithmetic;

template <class T> using is_std_integral = std::is_integral<T>;

template <class T> using is_empty = std::is_empty<T>;

//
// tuple_size, tuple_element
//
// @brief CuTe-local tuple-traits to prevent conflicts with other libraries.
// For cute:: types, we specialize std::tuple-traits, which is explicitly
// allowed.
//   tal::tuple, tal::array, tal::array_subbyte, etc
// But CuTe wants to treat some external types as tuples as well. For those,
// we specialize cute::tuple-traits to avoid polluting external traits.
//   dim3, uint3, etc

template <class T, class = void> struct tuple_size;

template <size_t I, class T, class = void> struct tuple_element;

template <size_t I, class T>
struct tuple_element<I, T, void_t<typename std::tuple_element<I, T>::type>>
    : std::tuple_element<I, T> {};

template <size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

} // namespace causalflow::tal