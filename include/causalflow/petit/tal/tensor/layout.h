#pragma once

#include "stride.h"

#include "causalflow/petit/tal/config.h"
#include "causalflow/petit/tal/detail/helper_macro.h"

namespace causalflow::tal {

// Aliases

template <class... Shapes> using Shape = tuple<Shapes...>;
template <class... Strides> using Stride = tuple<Strides...>;
template <class... Strides> using Step = tuple<Strides...>;
template <class... Coords> using Coord = tuple<Coords...>;
template <class... Layouts> using Tile = tuple<Layouts...>;

template <class... Ts>
TAL_HOST_DEVICE constexpr Shape<Ts...> make_shape(Ts const &...t) {
    return {t...};
}
template <class... Ts>
TAL_HOST_DEVICE constexpr Stride<Ts...> make_stride(Ts const &...t) {
    return {t...};
}
template <class... Ts>
TAL_HOST_DEVICE constexpr Step<Ts...> make_step(Ts const &...t) {
    return {t...};
}
template <class... Ts>
TAL_HOST_DEVICE constexpr Coord<Ts...> make_coord(Ts const &...t) {
    return {t...};
}
template <class... Ts>
TAL_HOST_DEVICE constexpr Tile<Ts...> make_tile(Ts const &...t) {
    return {t...};
}

template <class Shape, class Stride> class Layout {
  public:
    // NOTE: This defaults static Shapes/Strides correctly, but not dynamic
    TAL_HOST_DEVICE constexpr Layout(Shape const &shape = {},
                                     Stride const &stride = {})
        : shape_(shape), stride_(stride) {}

    // Map a logical coordinate to a linear index (Coord has no Underscore slice
    // operators) OR Slice the layout and return the sublayout (Coord has an
    // Underscore slice op)
    template <class Coord>
    TAL_HOST_DEVICE constexpr auto operator()(Coord const &coord) const {
        return crd2idx(coord, shape(), stride());
    }

    template <int... I> constexpr TAL_HOST_DEVICE auto shape() const {
        if constexpr (sizeof...(I) == 0) {
            return shape_;
        } else {
            return get<I...>(shape_);
        }
    }

    template <int... I> constexpr TAL_HOST_DEVICE auto stride() const {
        if constexpr (sizeof...(I) == 0) {
            return stride_;
        } else {
            return get<I...>(stride_);
        }
    }

  private:
    [[no_unique_address]] Shape shape_;
    [[no_unique_address]] Stride stride_;
};

template <class Layout> struct is_layout : false_type {};
template <class Shape, class Stride>
struct is_layout<Layout<Shape, Stride>> : true_type {};

//
// Layout construction
//

template <class Shape, class Stride>
TAL_HOST_DEVICE constexpr auto make_layout(Shape const &shape,
                                           Stride const &stride) {
    static_assert(is_tuple<Shape>::value || is_integral<Shape>::value);
    static_assert(is_tuple<Stride>::value || is_integral<Stride>::value);
    return Layout<Shape, Stride>(shape, stride);
}

template <int... Is, class Shape, class Stride>
TAL_HOST_DEVICE constexpr decltype(auto)
shape(Layout<Shape, Stride> const &layout) {
    return layout.template shape<Is...>();
}

// Return the number of elements in a mode
template <int... Is, class Shape, class Stride>
TAL_HOST_DEVICE constexpr auto size(Layout<Shape, Stride> const &layout) {
    return size(shape<Is...>(layout));
}

} // namespace causalflow::tal