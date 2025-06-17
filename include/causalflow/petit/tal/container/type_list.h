#pragma once

#include "causalflow/petit/tal/config.h"
#include "causalflow/petit/tal/numeric/integral_constant.h"
#include "causalflow/petit/tal/util/type_traits.h"

#include <tuple>

namespace causalflow::tal {

template <class... T> struct type_list {};

// get<I> for type_list<T...>
//   requires tuple_element_t<I,type_list<T...>> to have
//   std::is_default_constructible
template <size_t I, class... T>
TAL_HOST_DEVICE constexpr tuple_element_t<I, type_list<T...>>
get(type_list<T...> const &t) noexcept {
    return {};
}

template <class... T>
struct tuple_size<type_list<T...>> : integral_constant<size_t, sizeof...(T)> {};

template <size_t I, class... T> struct tuple_element<I, type_list<T...>> {
    using type = typename std::tuple_element<I, std::tuple<T...>>::type;
};

template <class... T>
struct tuple_size<const type_list<T...>>
    : integral_constant<size_t, sizeof...(T)> {};

template <size_t I, class... T> struct tuple_element<I, const type_list<T...>> {
    using type = typename std::tuple_element<I, std::tuple<T...>>::type;
};

} // namespace causalflow::tal