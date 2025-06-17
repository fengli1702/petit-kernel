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
#include "causalflow/petit/tal/numeric/math.h" // tal::max, tal::min

/** C++14 <functional> extensions */

namespace causalflow::tal {

/**************/
/** Identity **/
/**************/

struct identity {
    template <class T>
    TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {
        return static_cast<T &&>(arg);
    }
};

template <class R> struct constant_fn {
    template <class... T>
    TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&...) const {
        return r_;
    }
    R r_;
};

/***********/
/** Unary **/
/***********/

#define TAL_LEFT_UNARY_OP(NAME, OP)                                            \
    struct NAME {                                                              \
        template <class T>                                                     \
        TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {   \
            return OP static_cast<T &&>(arg);                                  \
        }                                                                      \
    }
#define TAL_RIGHT_UNARY_OP(NAME, OP)                                           \
    struct NAME {                                                              \
        template <class T>                                                     \
        TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {   \
            return static_cast<T &&>(arg) OP;                                  \
        }                                                                      \
    }
#define TAL_NAMED_UNARY_OP(NAME, OP)                                           \
    struct NAME {                                                              \
        template <class T>                                                     \
        TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {   \
            return OP(static_cast<T &&>(arg));                                 \
        }                                                                      \
    }

TAL_LEFT_UNARY_OP(unary_plus, +);
TAL_LEFT_UNARY_OP(negate, -);
TAL_LEFT_UNARY_OP(bit_not, ~);
TAL_LEFT_UNARY_OP(logical_not, !);
TAL_LEFT_UNARY_OP(dereference, *);
TAL_LEFT_UNARY_OP(address_of, &);
TAL_LEFT_UNARY_OP(pre_increment, ++);
TAL_LEFT_UNARY_OP(pre_decrement, --);

TAL_RIGHT_UNARY_OP(post_increment, ++);
TAL_RIGHT_UNARY_OP(post_decrement, --);

TAL_NAMED_UNARY_OP(abs_fn, abs);

#undef TAL_LEFT_UNARY_OP
#undef TAL_RIGHT_UNARY_OP
#undef TAL_NAMED_UNARY_OP

template <int Shift_> struct shift_right_const {
    static constexpr int Shift = Shift_;

    template <class T>
    TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {
        return static_cast<T &&>(arg) >> Shift;
    }
};

template <int Shift_> struct shift_left_const {
    static constexpr int Shift = Shift_;

    template <class T>
    TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) const {
        return static_cast<T &&>(arg) << Shift;
    }
};

/************/
/** Binary **/
/************/

#define TAL_BINARY_OP(NAME, OP)                                                \
    struct NAME {                                                              \
        template <class T, class U>                                            \
        TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&lhs,           \
                                                            U &&rhs) const {   \
            return static_cast<T &&>(lhs) OP static_cast<U &&>(rhs);           \
        }                                                                      \
    }
#define TAL_NAMED_BINARY_OP(NAME, OP)                                          \
    struct NAME {                                                              \
        template <class T, class U>                                            \
        TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&lhs,           \
                                                            U &&rhs) const {   \
            return OP(static_cast<T &&>(lhs), static_cast<U &&>(rhs));         \
        }                                                                      \
    }

TAL_BINARY_OP(plus, +);
TAL_BINARY_OP(minus, -);
TAL_BINARY_OP(multiplies, *);
TAL_BINARY_OP(divides, /);
TAL_BINARY_OP(modulus, %);

TAL_BINARY_OP(plus_assign, +=);
TAL_BINARY_OP(minus_assign, -=);
TAL_BINARY_OP(multiplies_assign, *=);
TAL_BINARY_OP(divides_assign, /=);
TAL_BINARY_OP(modulus_assign, %=);

TAL_BINARY_OP(bit_and, &);
TAL_BINARY_OP(bit_or, |);
TAL_BINARY_OP(bit_xor, ^);
TAL_BINARY_OP(left_shift, <<);
TAL_BINARY_OP(right_shift, >>);

TAL_BINARY_OP(bit_and_assign, &=);
TAL_BINARY_OP(bit_or_assign, |=);
TAL_BINARY_OP(bit_xor_assign, ^=);
TAL_BINARY_OP(left_shift_assign, <<=);
TAL_BINARY_OP(right_shift_assign, >>=);

TAL_BINARY_OP(logical_and, &&);
TAL_BINARY_OP(logical_or, ||);

TAL_BINARY_OP(equal_to, ==);
TAL_BINARY_OP(not_equal_to, !=);
TAL_BINARY_OP(greater, >);
TAL_BINARY_OP(less, <);
TAL_BINARY_OP(greater_equal, >=);
TAL_BINARY_OP(less_equal, <=);

TAL_NAMED_BINARY_OP(max_fn, tal::max);
TAL_NAMED_BINARY_OP(min_fn, tal::min);

#undef TAL_BINARY_OP
#undef TAL_NAMED_BINARY_OP

/**********/
/** Fold **/
/**********/

#define TAL_FOLD_OP(NAME, OP)                                                  \
    struct NAME##_unary_rfold {                                                \
        template <class... T>                                                  \
        TAL_HOST_DEVICE constexpr auto operator()(T &&...t) const {            \
            return (t OP...);                                                  \
        }                                                                      \
    };                                                                         \
    struct NAME##_unary_lfold {                                                \
        template <class... T>                                                  \
        TAL_HOST_DEVICE constexpr auto operator()(T &&...t) const {            \
            return (... OP t);                                                 \
        }                                                                      \
    };                                                                         \
    struct NAME##_binary_rfold {                                               \
        template <class U, class... T>                                         \
        TAL_HOST_DEVICE constexpr auto operator()(U &&u, T &&...t) const {     \
            return (t OP... OP u);                                             \
        }                                                                      \
    };                                                                         \
    struct NAME##_binary_lfold {                                               \
        template <class U, class... T>                                         \
        TAL_HOST_DEVICE constexpr auto operator()(U &&u, T &&...t) const {     \
            return (u OP... OP t);                                             \
        }                                                                      \
    }

TAL_FOLD_OP(plus, +);
TAL_FOLD_OP(minus, -);
TAL_FOLD_OP(multiplies, *);
TAL_FOLD_OP(divides, /);
TAL_FOLD_OP(modulus, %);

TAL_FOLD_OP(plus_assign, +=);
TAL_FOLD_OP(minus_assign, -=);
TAL_FOLD_OP(multiplies_assign, *=);
TAL_FOLD_OP(divides_assign, /=);
TAL_FOLD_OP(modulus_assign, %=);

TAL_FOLD_OP(bit_and, &);
TAL_FOLD_OP(bit_or, |);
TAL_FOLD_OP(bit_xor, ^);
TAL_FOLD_OP(left_shift, <<);
TAL_FOLD_OP(right_shift, >>);

TAL_FOLD_OP(bit_and_assign, &=);
TAL_FOLD_OP(bit_or_assign, |=);
TAL_FOLD_OP(bit_xor_assign, ^=);
TAL_FOLD_OP(left_shift_assign, <<=);
TAL_FOLD_OP(right_shift_assign, >>=);

TAL_FOLD_OP(logical_and, &&);
TAL_FOLD_OP(logical_or, ||);

TAL_FOLD_OP(equal_to, ==);
TAL_FOLD_OP(not_equal_to, !=);
TAL_FOLD_OP(greater, >);
TAL_FOLD_OP(less, <);
TAL_FOLD_OP(greater_equal, >=);
TAL_FOLD_OP(less_equal, <=);

#undef TAL_FOLD_OP

/**********/
/** Meta **/
/**********/

template <class Fn, class Arg> struct bound_fn {

    template <class T>
    TAL_HOST_DEVICE constexpr decltype(auto) operator()(T &&arg) {
        return fn_(arg_, static_cast<T &&>(arg));
    }

    Fn fn_;
    Arg arg_;
};

template <class Fn, class Arg>
TAL_HOST_DEVICE constexpr auto bind(Fn const &fn, Arg const &arg) {
    return bound_fn<Fn, Arg>{fn, arg};
}

} // namespace causalflow::tal