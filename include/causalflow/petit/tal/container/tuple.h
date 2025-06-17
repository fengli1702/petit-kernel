#pragma once

#include "causalflow/petit/tal/util/type_traits.h"
#include "packed_tuple.h"

namespace causalflow::tal {

template <class... Types> using tuple = packed_tuple<Types...>;

// For performance reasons, we want tuple to be trivially copyable.
static_assert(std::is_trivially_copyable<tuple<int, int>>::value, "");

//
// Custom is_tuple trait simply checks the existence of tuple_size
//      and assumes std::get<I>(.), std::tuple_element<I,.>
//
namespace detail {

template <class T>
auto has_tuple_size(T *) -> bool_constant<(0 <= tuple_size<T>::value)>;
auto has_tuple_size(...) -> false_type;

} // end namespace detail

template <class T> struct is_tuple : decltype(detail::has_tuple_size((T *)0)){};

template <typename T> constexpr bool is_tuple_v = is_tuple<T>::value;

//
// make_tuple (value-based implementation)
//

template <class... T>
TAL_HOST_DEVICE constexpr tuple<T...> make_tuple(T const &...t) {
    return {t...};
}

template <class... T>
struct tuple_size<const tuple<T...>>
    : std::integral_constant<size_t, sizeof...(T)> {};

template <size_t I, class... T>
struct tuple_element<I, const tuple<T...>>
    : std::tuple_element<I, const std::tuple<T...>> {};

} // namespace causalflow::tal
