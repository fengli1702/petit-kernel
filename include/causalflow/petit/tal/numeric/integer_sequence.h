#pragma once

#include "causalflow/petit/tal/util/type_traits.h"

namespace causalflow::tal {

template <int... Ints> using int_sequence = std::integer_sequence<int, Ints...>;

template <int N> using make_int_sequence = std::make_integer_sequence<int, N>;
template <int N> using make_seq = make_int_sequence<N>;

template <int... Ints> using seq = int_sequence<Ints...>;

template <class Tuple>
using tuple_seq = make_seq<tuple_size<std::remove_cvref_t<Tuple>>::value>;
} // namespace causalflow::tal