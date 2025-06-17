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
#include "causalflow/petit/tal/container/array.h"
#include "causalflow/petit/tal/pointer_base.h"
#include "layout.h"

namespace causalflow::tal {
//
// Engine -- owning or non-owning data store
//

// concept Engine {
//   using iterator     = ;
//   using value_type   = ;
//   using element_type = ;
//   using reference    = ;
//   iterator begin();
// };

template <class T, size_t N> struct ArrayEngine {
    using Storage = Array<T, N>;
    using Iterator = typename Storage::iterator;
    using ValueType = typename std::iterator_traits<Iterator>::value_type;

    Storage storage_;

    TAL_HOST_DEVICE constexpr Iterator const begin() const {
        return storage_.begin();
    }
    TAL_HOST_DEVICE constexpr Iterator begin() { return storage_.begin(); }
};

template <class Iterator_> struct ViewEngine {
    using Iterator = Iterator_;
    using ValueType = typename std::iterator_traits<Iterator>::value_type;

    Iterator storage_;

    TAL_HOST_DEVICE constexpr Iterator const begin() const { return storage_; }
    TAL_HOST_DEVICE constexpr Iterator begin() { return storage_; }
};

template <class Engine, class Layout> struct Tensor {
    using Iterator = typename Engine::Iterator;
    using ValueType = typename Engine::ValueType;

    TAL_HOST_DEVICE constexpr Tensor() {}

    TAL_HOST_DEVICE constexpr Tensor(Engine const &engine, Layout const &layout)
        : engine_(engine), layout_(layout) {}

    TAL_HOST_DEVICE inline Iterator data() { return engine_.begin(); }
    TAL_HOST_DEVICE inline Iterator const data() const {
        return engine_.begin();
    }

    template <class Coords>
    TAL_HOST_DEVICE inline ValueType &operator[](Coords const &coord) {
        return data()[layout()(coord)];
    }

    template <class Coords>
    TAL_HOST_DEVICE inline ValueType const &
    operator[](Coords const &coord) const {
        return data()[layout()(coord)];
    }

    TAL_HOST_DEVICE Layout constexpr const &layout() const { return layout_; }

    TAL_HOST_DEVICE constexpr decltype(auto) shape() const {
        return layout().shape();
    }

    TAL_HOST_DEVICE constexpr auto size() const { return size(layout()); }

    TAL_HOST_DEVICE constexpr decltype(auto) stride() const {
        return layout().stride();
    }

    Engine engine_;
    Layout layout_;
};

namespace detail {
// Customization point for creation of owning and non-owning Tensors
template <class T> struct MakeTensor {
    template <class Arg0, class... Args>
    TAL_HOST_DEVICE constexpr auto operator()(Arg0 const &arg0,
                                              Args const &...args) const {
        if constexpr (has_dereference<Arg0>::value) {
            // Construct a non-owning Tensor
            using Engine = ViewEngine<Arg0>;
            return Tensor{Engine{arg0}, args...};
        } else {
            // Construct an owning Tensor
            static_assert(
                (is_static<Arg0>::value && ... && is_static<Args>::value),
                "Dynamic owning tensors not supported");
            static_assert(sizeof...(Args) == 0 && is_layout<Arg0>::value,
                          "Must specify layout");
            using Layout = Arg0;
            // TODO: CuTE use co-size instead of size? Why?
            auto layout_size = size(Layout{});
            using Engine = ArrayEngine<T, layout_size>;
            return Tensor<Engine, Layout>();
        }
    }
};
} // namespace detail

// Return the number of elements in a mode
template <int... Is, class Engine, class Layout>
TAL_HOST_DEVICE constexpr auto size(Tensor<Engine, Layout> const &tensor) {
    return size<Is...>(tensor.layout());
}

//
// make_tensor
//

// Make an owning Tensor that will allocate a static array
// e.g. make_tensor<float>(Int<12>{})
template <class T, class... Args>
TAL_HOST_DEVICE constexpr auto make_tensor(Args const &...args) {
    static_assert((not has_dereference<Args>::value && ...),
                  "Expected layout args... in make_tensor<T>(args...)");
    return detail::MakeTensor<T>{}(args...);
}

// Make a non-owning Tensor that will use a pointer (view)
// e.g. make_tensor(vec.data(), 12)
template <class Iterator, class... Args>
TAL_HOST_DEVICE constexpr auto make_tensor(Iterator const &iter,
                                           Args const &...args) {
    static_assert(has_dereference<Iterator>::value,
                  "Expected iterator iter in make_tensor(iter, args...)");
    static_assert((not has_dereference<Args>::value && ...),
                  "Expected layout args... in make_tensor(iter, args...)");
    return detail::MakeTensor<Iterator>{}(iter, args...);
}

} // namespace causalflow::tal