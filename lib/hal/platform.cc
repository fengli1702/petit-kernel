#include "causalflow/petit/config.h"
#include "device.h"
#include <map>
#include <string>

namespace causalflow::petit::hal {

Device::~Device() {}
Platform::~Platform() {}

Platform *GetRocmPlatform();

static const std::map<std::string, std::function<Platform *()>> kPlatforms = {
#ifdef WITH_ROCM
    {"rocm", GetRocmPlatform},
#endif
};

Platform *GetPlatform(const char *backend) {
    std::string k(backend);
    auto it = kPlatforms.find(k);
    return it == kPlatforms.end() ? nullptr : it->second();
}

} // namespace causalflow::petit::hal