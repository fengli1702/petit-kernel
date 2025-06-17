#pragma once

#include <span>
#include <vector>

namespace causalflow::petit {

template <class T, class U>
static void FillRandomValue(T &op, std::vector<U> *data) {
    for (size_t i = 0; i < data->size(); ++i) {
        (*data)[i] = op();
    }
}

template <class T, class U, size_t kN>
static void FillRandomValue(T &op, std::span<U, kN> data) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = op();
    }
}
} // namespace causalflow::petit