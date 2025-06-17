#pragma once

#include "causalflow/petit/tal/numeric/integral_constant.h"
#include "causalflow/petit/tal/util/type_traits.h"

namespace causalflow::tal {
//
// has_dereference to determine if a type is an iterator concept
//

namespace detail {
template <class T, class = void> struct has_dereference : false_type {};
template <class T>
struct has_dereference<T, void_t<decltype(*declval<T &>())>> : true_type {};
} // end namespace detail

template <class T> using has_dereference = detail::has_dereference<T>;
} // namespace causalflow::tal