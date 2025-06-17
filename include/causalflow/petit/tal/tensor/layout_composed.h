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

/* This implements a ComposedLayout of the form
 *   LayoutA o Offset o LayoutB
 * and is useful in cases where composition() does not or cannot apply to
 * LayoutA and LayoutB. For example, when the "divisibility condition" in
 * shape_div is violated in composition(LayoutA, LayoutB).
 *
 * This ComposedLayout provides similar functionality to Layout including
 * tiling, partitioning, coordinate-to-index mapping and layout manipulations,
 * but is not considered a "normal" layout. For example, this layout provides
 * shape() and size() functions, but does not provide stride() functions.
 * Mostly, the similar functionality is accomplished by applying each operation
 * to LayoutB only as LayoutB defines the domain.
 */

#include "causalflow/petit/tal/tensor/layout.h"

namespace causalflow::tal {

// A Layout of non-trivially composable functions: F o I o L
template <class LayoutA, class Offset, class LayoutB> struct ComposedLayout {
    TAL_HOST_DEVICE constexpr ComposedLayout(LayoutA const &layoutA = {},
                                             Offset const &offset = {},
                                             LayoutB const &layoutB = {})
        : layout_a_(layoutA), offset_(offset), layout_b_(layoutB) {}

    //
    // Accessors
    //

    static constexpr int rank = LayoutB::rank;

    TAL_HOST_DEVICE constexpr decltype(auto) layout_a() const {
        return layout_a_;
    }

    TAL_HOST_DEVICE constexpr decltype(auto) offset() const { return offset_; }

    TAL_HOST_DEVICE constexpr decltype(auto) layout_b() const {
        return layout_b_;
    }

    TAL_HOST_DEVICE constexpr decltype(auto) layout() const { return *this; }

    TAL_HOST_DEVICE constexpr decltype(auto) shape() const {
        return layout_b().shape();
    }

    // Doesn't really make sense to ask for the strides of this "layout"
    TAL_HOST_DEVICE constexpr decltype(auto) stride() const = delete;

    //
    // Mappings
    //

    // Map a logical coordinate to a linear index (Coord has no Underscore slice
    // operators) OR Slice the layout and return the sublayout (Coord has an
    // Underscore slice op)
    template <class Coord>
    TAL_HOST_DEVICE constexpr auto operator()(Coord const &coord) const {
        return layout_a()(offset() + layout_b()(coord)); // (A o O o B)(c)
    }

    //
    // Compose
    //

    template <class OtherLayout>
    TAL_HOST_DEVICE constexpr auto compose(OtherLayout const &other) const {
        return composition(*this, other);
    }

    // Equality, return a static or dynamic boolean
    template <class... Args>
    TAL_HOST_DEVICE constexpr auto
    operator==(ComposedLayout<Args...> const &other) const {
        return this->layout_a() == other.layout_a() &&
               this->layout_b() == other.layout_b() &&
               this->offset() == other.offset();
    }

  private:
    [[no_unique_address]] LayoutA layout_a_;
    [[no_unique_address]] Offset offset_;
    [[no_unique_address]] LayoutB layout_b_;
};

template <class A, class O, class B>
struct is_layout<ComposedLayout<A, O, B>> : true_type {};

template <class T> struct is_composed_layout : false_type {};
template <class A, class O, class B>
struct is_composed_layout<ComposedLayout<A, O, B>> : true_type {};

//
// Constructors
//

template <class LayoutA, class Offset, class LayoutB>
TAL_HOST_DEVICE constexpr auto make_composed_layout(LayoutA const &layoutA,
                                                    Offset const &offset,
                                                    LayoutB const &layoutB) {
    return ComposedLayout<LayoutA, Offset, LayoutB>{layoutA, offset, layoutB};
}

// Return the shape of a mode
template <int... Is, class A, class O, class B>
TAL_HOST_DEVICE constexpr decltype(auto)
shape(ComposedLayout<A, O, B> const &layout) {
    return shape<Is...>(layout.layout_b());
}

// Doesn't make sense to directly ask for the strides of this "layout"
template <int... Is, class Fn, class O, class Layout>
TAL_HOST_DEVICE constexpr decltype(auto)
stride(ComposedLayout<Fn, O, Layout> const &layout) = delete;

// Return the number of elements in a mode
template <int... Is, class A, class O, class B>
TAL_HOST_DEVICE constexpr decltype(auto)
size(ComposedLayout<A, O, B> const &layout) {
    return size<Is...>(layout.layout_b());
}

} // namespace causalflow::tal