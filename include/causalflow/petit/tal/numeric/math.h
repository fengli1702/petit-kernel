#pragma once

#include "causalflow/petit/tal/config.h"
#include "causalflow/petit/tal/util/type_traits.h"

namespace causalflow::tal {

//
// Common Operations
//

template <class T, class U,
          __TAL_REQUIRES(is_arithmetic<T>::value &&is_arithmetic<U>::value)>
TAL_HOST_DEVICE constexpr auto max(T const &t, U const &u) {
    return t < u ? u : t;
}

template <class T, class U,
          __TAL_REQUIRES(is_arithmetic<T>::value &&is_arithmetic<U>::value)>
TAL_HOST_DEVICE constexpr auto min(T const &t, U const &u) {
    return t < u ? t : u;
}

template <class IntDiv, class IntMod> struct DivModReturnType {
    IntDiv div_;
    IntMod mod_;
    TAL_HOST_DEVICE constexpr DivModReturnType(IntDiv const &div,
                                               IntMod const &mod)
        : div_(div), mod_(mod) {}
};

// General divmod
template <class CInt0, class CInt1>
TAL_HOST_DEVICE constexpr auto divmod(CInt0 const &a, CInt1 const &b) {
    return DivModReturnType{a / b, a % b};
}
} // namespace causalflow::tal