#pragma once

#include <fstream>
#include <string>
#include <vector>

namespace causalflow::petit::tests {

template <class T>
[[maybe_unused]] static inline void WriteBinary(const std::string &path,
                                                const std::vector<T> &data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()),
               data.size() * sizeof(T));
}

} // end namespace causalflow::petit::tests