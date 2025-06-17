/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "causalflow/petit/tal/config.h"
#include "causalflow/petit/tal/container/tuple.h"
#include "causalflow/petit/tal/int_tuple.h"
#include "causalflow/petit/tal/numeric/integer_sequence.h"
#include "causalflow/petit/tal/numeric/integral_constant.h"
#include "causalflow/petit/tal/util/type_traits.h"

namespace causalflow::tal {

/** crd2idx(c,s,d) maps a coordinate within <Shape,Stride> to an index
 *
 * This is computed as follows:
 *  [coord, shape, and stride are all integers => step forward by stride]
 * op(c, s, d)             => c * d
 *  [coord is integer, shape and stride are tuple => divmod coord for each mode]
 * op(c, (s,S), (d,D))     => op(c % prod(s), s, d) + op(c / prod(s), (S), (D))
 *  [coord, shape, and stride are all tuples => consider each mode
 * independently] op((c,C), (s,S), (d,D)) => op(c, s, d) + op((C), (S), (D))
 */
template <class Coord, class Shape, class Stride>
TAL_HOST_DEVICE constexpr auto crd2idx(Coord const &coord, Shape const &shape,
                                       Stride const &stride);

namespace detail {

template <class Coord, class Shape, class Stride, int... Is>
TAL_HOST_DEVICE constexpr auto crd2idx_ttt(Coord const &coord,
                                           Shape const &shape,
                                           Stride const &stride, seq<Is...>) {
    return (... + crd2idx(get<Is>(coord), get<Is>(shape), get<Is>(stride)));
}

template <class CInt, class STuple, class DTuple, int I0, int... Is>
TAL_HOST_DEVICE constexpr auto
crd2idx_itt(CInt const &coord, STuple const &shape, DTuple const &stride,
            seq<I0, Is...>) {
    if constexpr (sizeof...(Is) ==
                  0) { // Avoid recursion and mod on single/last iter
        return crd2idx(coord, get<I0>(shape), get<I0>(stride));
    } else if constexpr (is_constant<0, CInt>::value) {
        return crd2idx(_0{}, get<I0>(shape), get<I0>(stride)) +
               (_0{} + ... + crd2idx(_0{}, get<Is>(shape), get<Is>(stride)));
    } else { // General case
        auto [div, mod] = divmod(coord, product(get<I0>(shape)));
        return crd2idx(mod, get<I0>(shape), get<I0>(stride)) +
               crd2idx_itt(div, shape, stride, seq<Is...>{});
    }

    unreachable();
}

} // end namespace detail

template <class Coord, class Shape, class Stride>
TAL_HOST_DEVICE constexpr auto crd2idx(Coord const &coord, Shape const &shape,
                                       Stride const &stride) {
    if constexpr (is_tuple<Coord>::value) {
        if constexpr (is_tuple<Shape>::value) { // tuple tuple tuple
            static_assert(tuple_size<Coord>::value == tuple_size<Shape>::value,
                          "Mismatched Ranks");
            static_assert(tuple_size<Coord>::value == tuple_size<Stride>::value,
                          "Mismatched Ranks");
            return detail::crd2idx_ttt(coord, shape, stride,
                                       tuple_seq<Coord>{});
        } else { // tuple "int" "int"
            static_assert(sizeof(Coord) == 0, "Invalid parameters");
        }
    } else {
        if constexpr (is_tuple<Shape>::value) { // "int" tuple tuple
            static_assert(tuple_size<Shape>::value == tuple_size<Stride>::value,
                          "Mismatched Ranks");
            return detail::crd2idx_itt(coord, shape, stride,
                                       tuple_seq<Shape>{});
        } else { // "int" "int" "int"
            return coord * stride;
        }
    }

    unreachable();
}

} // namespace causalflow::tal