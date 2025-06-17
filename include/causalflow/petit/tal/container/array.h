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
#include "causalflow/petit/tal/numeric/integral_constant.h"
#include "causalflow/petit/tal/util/type_traits.h"

namespace causalflow::tal {

template <class T, size_t N> struct Array {
    using element_type = T;
    using value_type = remove_cv_t<T>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = element_type &;
    using const_reference = const element_type &;
    using pointer = element_type *;
    using const_pointer = const element_type *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    TAL_HOST_DEVICE constexpr reference operator[](size_type pos) {
        return begin()[pos];
    }

    TAL_HOST_DEVICE constexpr const_reference operator[](size_type pos) const {
        return begin()[pos];
    }

    TAL_HOST_DEVICE constexpr reference front() { return *begin(); }

    TAL_HOST_DEVICE constexpr const_reference front() const { return *begin(); }

    TAL_HOST_DEVICE constexpr reference back() {
        // return *rbegin();
        return operator[](N - 1);
    }

    TAL_HOST_DEVICE constexpr const_reference back() const {
        // return *rbegin();
        return operator[](N - 1);
    }

    TAL_HOST_DEVICE constexpr T *data() { return __elems_; }

    TAL_HOST_DEVICE constexpr T const *data() const { return __elems_; }

    TAL_HOST_DEVICE constexpr iterator begin() { return data(); }

    TAL_HOST_DEVICE constexpr const_iterator begin() const { return data(); }

    TAL_HOST_DEVICE constexpr const_iterator cbegin() { return begin(); }

    TAL_HOST_DEVICE constexpr const_iterator cbegin() const { return begin(); }

    TAL_HOST_DEVICE constexpr iterator end() { return data() + size(); }

    TAL_HOST_DEVICE constexpr const_iterator end() const {
        return data() + size();
    }

    TAL_HOST_DEVICE constexpr const_iterator cend() { return end(); }

    TAL_HOST_DEVICE constexpr const_iterator cend() const { return end(); }

    TAL_HOST_DEVICE constexpr bool empty() const { return size() == 0; }

    TAL_HOST_DEVICE constexpr size_type size() const { return N; }

    TAL_HOST_DEVICE constexpr size_type max_size() const { return size(); }

    TAL_HOST_DEVICE constexpr void fill(const T &value) {
        for (auto &e : *this) {
            e = value;
        }
    }

    TAL_HOST_DEVICE constexpr void clear() { fill(T(0)); }

    element_type __elems_[N];
};

template <class T> struct Array<T, 0> {
    using element_type = T;
    using value_type = remove_cv_t<T>;
    using size_type = size_t;
    using difference_type = ptrdiff_t;
    using reference = element_type &;
    using const_reference = const element_type &;
    using pointer = element_type *;
    using const_pointer = const element_type *;
    using const_iterator = const_pointer;
    using iterator = pointer;

    TAL_HOST_DEVICE constexpr reference operator[](size_type pos) {
        return begin()[pos];
    }

    TAL_HOST_DEVICE constexpr const_reference operator[](size_type pos) const {
        return begin()[pos];
    }

    TAL_HOST_DEVICE constexpr reference front() { return *begin(); }

    TAL_HOST_DEVICE constexpr const_reference front() const { return *begin(); }

    TAL_HOST_DEVICE constexpr reference back() { return *begin(); }

    TAL_HOST_DEVICE constexpr const_reference back() const { return *begin(); }

    TAL_HOST_DEVICE constexpr T *data() { return nullptr; }

    TAL_HOST_DEVICE constexpr T const *data() const { return nullptr; }

    TAL_HOST_DEVICE constexpr iterator begin() { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator begin() const { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator cbegin() { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator cbegin() const { return nullptr; }

    TAL_HOST_DEVICE constexpr iterator end() { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator end() const { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator cend() { return nullptr; }

    TAL_HOST_DEVICE constexpr const_iterator cend() const { return nullptr; }

    TAL_HOST_DEVICE constexpr bool empty() const { return true; }

    TAL_HOST_DEVICE constexpr size_type size() const { return 0; }

    TAL_HOST_DEVICE constexpr size_type max_size() const { return 0; }

    TAL_HOST_DEVICE constexpr void fill(const T &value) {}

    TAL_HOST_DEVICE constexpr void clear() {}

    TAL_HOST_DEVICE constexpr void swap(Array &other) {}
};

template <class T, size_t N>
TAL_HOST_DEVICE constexpr bool operator==(Array<T, N> const &lhs,
                                          Array<T, N> const &rhs) {
    for (size_t i = 0; i < N; ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }
    return true;
}

template <size_t I, class T, size_t N>
TAL_HOST_DEVICE constexpr T &get(Array<T, N> &a) {
    static_assert(I < N, "Index out of range");
    return a[I];
}

template <size_t I, class T, size_t N>
TAL_HOST_DEVICE constexpr T const &get(Array<T, N> const &a) {
    static_assert(I < N, "Index out of range");
    return a[I];
}

template <size_t I, class T, size_t N>
TAL_HOST_DEVICE constexpr T &&get(Array<T, N> &&a) {
    static_assert(I < N, "Index out of range");
    return std::move(a[I]);
}

template <class T, size_t N>
struct tuple_size<tal::Array<T, N>> : integral_constant<size_t, N> {};

template <size_t I, class T, size_t N>
struct tuple_element<I, tal::Array<T, N>> {
    using type = T;
};

template <class T, size_t N>
struct tuple_size<tal::Array<T, N> const> : integral_constant<size_t, N> {};

template <size_t I, class T, size_t N>
struct tuple_element<I, tal::Array<T, N> const> {
    using type = T;
};

} // namespace causalflow::tal
