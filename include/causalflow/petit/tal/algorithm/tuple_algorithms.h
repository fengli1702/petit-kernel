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
#include "causalflow/petit/tal/numeric/integer_sequence.h"
#include "causalflow/petit/tal/numeric/integral_constant.h"
#include "causalflow/petit/tal/util/type_traits.h"

/// @file tuple_algorithms.hpp
/// @brief Common algorithms on (hierarchical) tuples
///
/// Code guidelines and style preferences:
///
/// For perfect forwarding, don't use std::forward, because it may not
/// be defined in device code when compiling with NVRTC. Instead, use
/// `static_cast<ParameterType&&>(parameter_name)`.
///
/// CuTe generally does not bother forwarding functions, as
/// reference-qualified member functions are rare in this code base.
///
/// Throughout CUTLASS, tal::make_tuple always needs to be called
/// namespace-qualified, EVEN If inside the cute namespace and/or in
/// scope of a "using namespace cute" declaration. Otherwise, the
/// compiler may select std::make_tuple instead of tal::make_tuple,
/// due to argument-dependent lookup.

namespace causalflow::tal {

//
// Apply (Unpack)
// (t, f) => f(t_0,t_1,...,t_n)
//

namespace detail {

template <class T, class F, int... I>
TAL_HOST_DEVICE constexpr auto apply(T &&t, F &&f, seq<I...>) {
    return f(get<I>(static_cast<T &&>(t))...);
}

} // end namespace detail

template <class T, class F> TAL_HOST_DEVICE constexpr auto apply(T &&t, F &&f) {
    return detail::apply(static_cast<T &&>(t), f, tuple_seq<T>{});
}

//
// Transform Apply
// (t, f, g) => g(f(t_0),f(t_1),...)
//

namespace detail {

template <class T, class F, class G, int... I>
TAL_HOST_DEVICE constexpr auto tapply(T &&t, F &&f, G &&g, seq<I...>) {
    return g(f(get<I>(static_cast<T &&>(t)))...);
}

template <class T0, class T1, class F, class G, int... I>
TAL_HOST_DEVICE constexpr auto tapply(T0 &&t0, T1 &&t1, F &&f, G &&g,
                                      seq<I...>) {
    return g(
        f(get<I>(static_cast<T0 &&>(t0)), get<I>(static_cast<T1 &&>(t1)))...);
}

template <class T0, class T1, class T2, class F, class G, int... I>
TAL_HOST_DEVICE constexpr auto tapply(T0 &&t0, T1 &&t1, T2 &&t2, F &&f, G &&g,
                                      seq<I...>) {
    return g(f(get<I>(static_cast<T0 &&>(t0)), get<I>(static_cast<T1 &&>(t1)),
               get<I>(static_cast<T2 &&>(t2)))...);
}

} // end namespace detail

template <class T, class F, class G>
TAL_HOST_DEVICE constexpr auto transform_apply(T &&t, F &&f, G &&g) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::tapply(static_cast<T &&>(t), f, g, tuple_seq<T>{});
    } else {
        return g(f(static_cast<T &&>(t)));
    }

    unreachable();
}

template <class T0, class T1, class F, class G>
TAL_HOST_DEVICE constexpr auto transform_apply(T0 &&t0, T1 &&t1, F &&f, G &&g) {
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        return detail::tapply(static_cast<T0 &&>(t0), static_cast<T1 &&>(t1), f,
                              g, tuple_seq<T0>{});
    } else {
        return g(f(static_cast<T0 &&>(t0), static_cast<T1 &&>(t1)));
    }

    unreachable();
}

template <class T0, class T1, class T2, class F, class G>
TAL_HOST_DEVICE constexpr auto transform_apply(T0 &&t0, T1 &&t1, T2 &&t2, F &&f,
                                               G &&g) {
    if constexpr (is_tuple<remove_cvref_t<T0>>::value) {
        return detail::tapply(static_cast<T0 &&>(t0), static_cast<T1 &&>(t1),
                              static_cast<T2 &&>(t2), f, g, tuple_seq<T0>{});
    } else {
        return g(f(static_cast<T0 &&>(t0), static_cast<T1 &&>(t1),
                   static_cast<T2 &&>(t2)));
    }

    unreachable();
}

//
// For Each
// (t, f) => f(t_0),f(t_1),...,f(t_n)
//

template <class T, class F>
TAL_HOST_DEVICE constexpr void for_each(T &&t, F &&f) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::apply(
            t, [&](auto &&...a) { (f(static_cast<decltype(a) &&>(a)), ...); },
            tuple_seq<T>{});
    } else {
        return f(static_cast<T &&>(t));
    }

    unreachable();
}

template <class T, class F>
TAL_HOST_DEVICE constexpr auto for_each_leaf(T &&t, F &&f) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::apply(
            static_cast<T &&>(t),
            [&](auto &&...a) {
                return (for_each_leaf(static_cast<decltype(a) &&>(a), f), ...);
            },
            tuple_seq<T>{});
    } else {
        return f(static_cast<T &&>(t));
    }

    unreachable();
}

//
// Transform
// (t, f) => (f(t_0),f(t_1),...,f(t_n))
//

template <class T, class F>
TAL_HOST_DEVICE constexpr auto transform(T const &t, F &&f) {
    if constexpr (is_tuple<T>::value) {
        return detail::tapply(
            t, f, [](auto const &...a) { return tal::make_tuple(a...); },
            tuple_seq<T>{});
    } else {
        return f(t);
    }

    unreachable();
}

template <class T0, class T1, class F>
TAL_HOST_DEVICE constexpr auto transform(T0 const &t0, T1 const &t1, F &&f) {
    if constexpr (is_tuple<T0>::value) {
        static_assert(tuple_size<T0>::value == tuple_size<T1>::value,
                      "Mismatched tuple_size");
        return detail::tapply(
            t0, t1, f, [](auto const &...a) { return tal::make_tuple(a...); },
            tuple_seq<T0>{});
    } else {
        return f(t0, t1);
    }

    unreachable();
}

template <class T0, class T1, class T2, class F>
TAL_HOST_DEVICE constexpr auto transform(T0 const &t0, T1 const &t1,
                                         T2 const &t2, F &&f) {
    if constexpr (is_tuple<T0>::value) {
        static_assert(tuple_size<T0>::value == tuple_size<T1>::value,
                      "Mismatched tuple_size");
        static_assert(tuple_size<T0>::value == tuple_size<T2>::value,
                      "Mismatched tuple_size");
        return detail::tapply(
            t0, t1, t2, f,
            [](auto const &...a) { return tal::make_tuple(a...); },
            tuple_seq<T0>{});
    } else {
        return f(t0, t1, t2);
    }

    unreachable();
}

template <class T, class F>
TAL_HOST_DEVICE constexpr auto transform_leaf(T const &t, F &&f) {
    if constexpr (is_tuple<T>::value) {
        return transform(t,
                         [&](auto const &a) { return transform_leaf(a, f); });
    } else {
        return f(t);
    }

    unreachable();
}

template <class T0, class T1, class F>
TAL_HOST_DEVICE constexpr auto transform_leaf(T0 const &t0, T1 const &t1,
                                              F &&f) {
    if constexpr (is_tuple<T0>::value) {
        return transform(t0, t1, [&](auto const &a, auto const &b) {
            return transform_leaf(a, b, f);
        });
    } else {
        return f(t0, t1);
    }

    unreachable();
}

//
// find and find_if
//

namespace detail {

template <class T, class F, int I, int... Is>
TAL_HOST_DEVICE constexpr auto find_if(T const &t, F &&f, seq<I, Is...>) {
    if constexpr (decltype(f(get<I>(t)))::value) {
        return tal::C<I>{};
    } else if constexpr (sizeof...(Is) == 0) {
        return tal::C<I + 1>{};
    } else {
        return find_if(t, f, seq<Is...>{});
    }

    unreachable();
}

} // end namespace detail

template <class T, class F>
TAL_HOST_DEVICE constexpr auto find_if(T const &t, F &&f) {
    if constexpr (is_tuple<T>::value) {
        return detail::find_if(t, f, tuple_seq<T>{});
    } else {
        return tal::C < decltype(f(t))::value ? 0 : 1 > {};
    }

    unreachable();
}

template <class T, class X>
TAL_HOST_DEVICE constexpr auto find(T const &t, X const &x) {
    return find_if(t, [&](auto const &v) {
        return v == x;
    }); // This should always return a static true/false
}

template <class T, class F>
TAL_HOST_DEVICE constexpr auto any_of(T const &t, F &&f) {
    if constexpr (is_tuple<T>::value) {
        return detail::apply(
            tal::transform(t, f),
            [&](auto const &...a) { return (false_type{} || ... || a); },
            tuple_seq<T>{});
    } else {
        return f(t);
    }

    unreachable();
}

template <class T, class F>
TAL_HOST_DEVICE constexpr auto all_of(T const &t, F &&f) {
    if constexpr (is_tuple<T>::value) {
        return detail::apply(
            tal::transform(t, f),
            [&](auto const &...a) { return (true_type{} && ... && a); },
            tuple_seq<T>{});
    } else {
        return f(t);
    }

    unreachable();
}

template <class T, class F>
TAL_HOST_DEVICE constexpr auto none_of(T const &t, F &&f) {
    return not any_of(t, f);
}

//
// Filter
// (t, f) => <f(t_0),f(t_1),...,f(t_n)>
//

//
// Fold (Reduce, Accumulate)
// (t, v, f) => f(...f(f(v,t_0),t_1),...,t_n)
//

namespace detail {

template <class Fn, class Val> struct FoldAdaptor {
    template <class X> TAL_HOST_DEVICE constexpr auto operator|(X &&x) {
        auto r = fn_(val_, static_cast<X &&>(x));
        return FoldAdaptor<Fn, decltype(r)>{fn_, r};
    }
    Fn fn_;
    Val val_;
};

template <class T, class V, class F, int... Is>
TAL_HOST_DEVICE constexpr auto fold(T &&t, V const &v, F &&f, seq<Is...>) {
    return (FoldAdaptor<F, V>{f, v} | ... | get<Is>(static_cast<T &&>(t))).val_;
}

} // end namespace detail

template <class T, class V, class F>
TAL_HOST_DEVICE constexpr auto fold(T &&t, V const &v, F &&f) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return detail::fold(static_cast<T &&>(t), v, f, tuple_seq<T>{});
    } else {
        return f(v, static_cast<T &&>(t));
    }

    unreachable();
}

//
// front, back, take, select, unwrap
//

// Get the first non-tuple element in a hierarchical tuple
template <class T> TAL_HOST_DEVICE constexpr decltype(auto) front(T &&t) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        return front(get<0>(static_cast<T &&>(t)));
    } else {
        return static_cast<T &&>(t);
    }

    unreachable();
}

// Get the last non-tuple element in a hierarchical tuple
template <class T> TAL_HOST_DEVICE constexpr decltype(auto) back(T &&t) {
    if constexpr (is_tuple<remove_cvref_t<T>>::value) {
        constexpr int N = tuple_size<remove_cvref_t<T>>::value;

        // MSVC needs a bit of extra help here deducing return types.
        // We help it by peeling off the nonrecursive case a level "early."
        if constexpr (!is_tuple<remove_cvref_t<decltype(get<N - 1>(
                          static_cast<T &&>(t)))>>::value) {
            return get<N - 1>(static_cast<T &&>(t));
        } else {
            return back(get<N - 1>(static_cast<T &&>(t)));
        }
    } else {
        return static_cast<T &&>(t);
    }

    unreachable();
}

// Select tuple elements with given indices.
template <int... I, class T> TAL_HOST_DEVICE constexpr auto select(T const &t) {
    return tal::make_tuple(get<I>(t)...);
}

// Wrap non-tuples into rank-1 tuples or forward
template <class T> TAL_HOST_DEVICE constexpr auto wrap(T const &t) {
    if constexpr (is_tuple<T>::value) {
        return t;
    } else {
        return tal::make_tuple(t);
    }

    unreachable();
}

// Unwrap rank-1 tuples until we're left with a rank>1 tuple or a non-tuple
template <class T> TAL_HOST_DEVICE constexpr auto unwrap(T const &t) {
    if constexpr (is_tuple<T>::value) {
        if constexpr (tuple_size<T>::value == 1) {
            return unwrap(get<0>(t));
        } else {
            return t;
        }
    } else {
        return t;
    }

    unreachable();
}

//
// Flatten and Unflatten
//

template <class T> struct is_flat : true_type {};

template <class... Ts>
struct is_flat<tuple<Ts...>>
    : bool_constant<(true && ... && (not is_tuple<Ts>::value))> {};

// Flatten a hierarchical tuple to a tuple of depth one
//   and wrap non-tuples into a rank-1 tuple.
template <class T> TAL_HOST_DEVICE constexpr auto flatten_to_tuple(T const &t) {
    if constexpr (is_tuple<T>::value) {
        if constexpr (is_flat<T>::value) { // Shortcut for perf
            return t;
        } else {
            return filter_tuple(
                t, [](auto const &a) { return flatten_to_tuple(a); });
        }
    } else {
        return tal::make_tuple(t);
    }

    unreachable();
}

// Flatten a hierarchical tuple to a tuple of depth one
//   and leave non-tuple untouched.
template <class T> TAL_HOST_DEVICE constexpr auto flatten(T const &t) {
    if constexpr (is_tuple<T>::value) {
        if constexpr (is_flat<T>::value) { // Shortcut for perf
            return t;
        } else {
            return filter_tuple(
                t, [](auto const &a) { return flatten_to_tuple(a); });
        }
    } else {
        return t;
    }

    unreachable();
}

namespace detail {

template <class FlatTuple, class TargetProfile>
TAL_HOST_DEVICE constexpr auto
unflatten_impl(FlatTuple const &flat_tuple,
               TargetProfile const &target_profile) {
    if constexpr (is_tuple<TargetProfile>::value) {
        return fold(
            target_profile, tal::make_tuple(tal::make_tuple(), flat_tuple),
            [](auto const &v, auto const &t) {
                auto [result, remaining_tuple] = v;
                auto [sub_result, sub_tuple] =
                    unflatten_impl(remaining_tuple, t);
                return tal::make_tuple(append(result, sub_result), sub_tuple);
            });
    } else {
        return tal::make_tuple(
            get<0>(flat_tuple),
            take<1, decltype(rank(flat_tuple))::value>(flat_tuple));
    }

    unreachable();
}

} // end namespace detail

// Unflatten a flat tuple into a hierarchical tuple
// @pre flatten(@a flat_tuple) == @a flat_tuple
// @pre rank(flatten(@a target_profile)) == rank(@a flat_tuple)
// @post congruent(@a result, @a target_profile)
// @post flatten(@a result) == @a flat_tuple
template <class FlatTuple, class TargetProfile>
TAL_HOST_DEVICE constexpr auto unflatten(FlatTuple const &flat_tuple,
                                         TargetProfile const &target_profile) {
    auto [unflatten_tuple, flat_remainder] =
        detail::unflatten_impl(flat_tuple, target_profile);
    CUTE_STATIC_ASSERT_V(rank(flat_remainder) == Int<0>{});
    return unflatten_tuple;
}

//
// insert and remove and replace
//

namespace detail {

// Shortcut around tal::tuple_cat for common insert/remove/repeat cases
template <class T, class X, int... I, int... J, int... K>
TAL_HOST_DEVICE constexpr auto construct(T const &t, X const &x, seq<I...>,
                                         seq<J...>, seq<K...>) {
    return tal::make_tuple(get<I>(t)..., (void(J), x)..., get<K>(t)...);
}

} // end namespace detail

// Replace the last element of the tuple with x
template <class T, class X>
TAL_HOST_DEVICE constexpr auto replace_back(T const &t, X const &x) {
    if constexpr (is_tuple<T>::value) {
        return detail::construct(t, x, make_seq<tuple_size<T>::value - 1>{},
                                 seq<0>{}, seq<>{});
    } else {
        return x;
    }

    unreachable();
}

//
// Make a tuple of Xs of tuple_size N
//

template <int N, class X>
TAL_HOST_DEVICE constexpr auto tuple_repeat(X const &x) {
    return detail::construct(0, x, seq<>{}, make_seq<N>{}, seq<>{});
}

//
// Make repeated Xs of rank N
//

template <int N, class X> TAL_HOST_DEVICE constexpr auto repeat(X const &x) {
    if constexpr (N == 1) {
        return x;
    } else {
        return detail::construct(0, x, seq<>{}, make_seq<N>{}, seq<>{});
    }

    unreachable();
}

//
// Make a tuple of Xs the same profile as tuple T
//

template <class T, class X>
TAL_HOST_DEVICE constexpr auto repeat_like(T const &t, X const &x) {
    if constexpr (is_tuple<T>::value) {
        return transform(t, [&](auto const &a) { return repeat_like(a, x); });
    } else {
        return x;
    }

    unreachable();
}

//
// Extend a T to rank N by appending/prepending an element
//

template <int N, class T, class X>
TAL_HOST_DEVICE constexpr auto append(T const &a, X const &x) {
    if constexpr (is_tuple<T>::value) {
        if constexpr (N == tuple_size<T>::value) {
            return a;
        } else {
            static_assert(N > tuple_size<T>::value);
            return detail::construct(a, x, make_seq<tuple_size<T>::value>{},
                                     make_seq<N - tuple_size<T>::value>{},
                                     seq<>{});
        }
    } else {
        if constexpr (N == 1) {
            return a;
        } else {
            return detail::construct(tal::make_tuple(a), x, seq<0>{},
                                     make_seq<N - 1>{}, seq<>{});
        }
    }

    unreachable();
}

template <class T, class X>
TAL_HOST_DEVICE constexpr auto append(T const &a, X const &x) {
    if constexpr (is_tuple<T>::value) {
        return detail::construct(a, x, make_seq<tuple_size<T>::value>{},
                                 seq<0>{}, seq<>{});
    } else {
        return tal::make_tuple(a, x);
    }

    unreachable();
}

template <int N, class T, class X>
TAL_HOST_DEVICE constexpr auto prepend(T const &a, X const &x) {
    if constexpr (is_tuple<T>::value) {
        if constexpr (N == tuple_size<T>::value) {
            return a;
        } else {
            static_assert(N > tuple_size<T>::value);
            return detail::construct(a, x, seq<>{},
                                     make_seq<N - tuple_size<T>::value>{},
                                     make_seq<tuple_size<T>::value>{});
        }
    } else {
        if constexpr (N == 1) {
            return a;
        } else {
            static_assert(N > 1);
            return detail::construct(tal::make_tuple(a), x, seq<>{},
                                     make_seq<N - 1>{}, seq<0>{});
        }
    }

    unreachable();
}

template <class T, class X>
TAL_HOST_DEVICE constexpr auto prepend(T const &a, X const &x) {
    if constexpr (is_tuple<T>::value) {
        return detail::construct(a, x, seq<>{}, seq<0>{},
                                 make_seq<tuple_size<T>::value>{});
    } else {
        return tal::make_tuple(x, a);
    }

    unreachable();
}

//
// Inclusive scan (prefix sum)
//

namespace detail {

template <class T, class V, class F, int I, int... Is>
TAL_HOST_DEVICE constexpr auto iscan(T const &t, V const &v, F &&f,
                                     seq<I, Is...>) {
    // Apply the function to v and the element at I
    auto v_next = f(v, get<I>(t));
    // Replace I with v_next
    auto t_next = replace<I>(t, v_next);

#if 0
  std::cout << "ISCAN i" << I << std::endl;
  std::cout << "  t      " << t << std::endl;
  std::cout << "  i      " << v << std::endl;
  std::cout << "  f(i,t) " << v_next << std::endl;
  std::cout << "  t_n    " << t_next << std::endl;
#endif

    if constexpr (sizeof...(Is) == 0) {
        return t_next;
    } else {
        return iscan(t_next, v_next, f, seq<Is...>{});
    }

    unreachable();
}

} // end namespace detail

template <class T, class V, class F>
TAL_HOST_DEVICE constexpr auto iscan(T const &t, V const &v, F &&f) {
    return detail::iscan(t, v, f, tuple_seq<T>{});
}

//
// Exclusive scan (prefix sum)
//

namespace detail {

template <class T, class V, class F, int I, int... Is>
TAL_HOST_DEVICE constexpr auto escan(T const &t, V const &v, F &&f,
                                     seq<I, Is...>) {
    if constexpr (sizeof...(Is) == 0) {
        // Replace I with v
        return replace<I>(t, v);
    } else {
        // Apply the function to v and the element at I
        auto v_next = f(v, get<I>(t));
        // Replace I with v
        auto t_next = replace<I>(t, v);

#if 0
    std::cout << "ESCAN i" << I << std::endl;
    std::cout << "  t      " << t << std::endl;
    std::cout << "  i      " << v << std::endl;
    std::cout << "  f(i,t) " << v_next << std::endl;
    std::cout << "  t_n    " << t_next << std::endl;
#endif

        // Recurse
        return escan(t_next, v_next, f, seq<Is...>{});
    }

    unreachable();
}

} // end namespace detail

template <class T, class V, class F>
TAL_HOST_DEVICE constexpr auto escan(T const &t, V const &v, F &&f) {
    return detail::escan(t, v, f, tuple_seq<T>{});
}

//
// Zip (Transpose)
//

// Take       ((a,b,c,...),(x,y,z,...),...)        rank-R0 x rank-R1 input
// to produce ((a,x,...),(b,y,...),(c,z,...),...)  rank-R1 x rank-R0 output

} // namespace causalflow::tal