#pragma once

#include <absl/status/status.h>
#include <memory>

namespace causalflow::petit::hal {

class Device {
  public:
    virtual absl::Status Malloc(void **ptr, size_t size) = 0;
    virtual absl::Status Free(void *ptr) = 0;
    virtual absl::Status Memset(void *ptr, int value, size_t size) = 0;
    virtual absl::Status CopyToDevice(void *dst, const void *src,
                                      size_t size) = 0;
    virtual absl::Status CopyToHost(void *dst, const void *src,
                                    size_t size) = 0;
    virtual absl::Status Synchronize() = 0;
    virtual ~Device();

  protected:
    Device() = default;
    ;
};

class Platform {
  public:
    virtual absl::Status GetDevice(int id, std::unique_ptr<Device> *result) = 0;
    virtual ~Platform();

  protected:
    Platform() = default;
};

Platform *GetPlatform(const char *);

} // namespace causalflow::petit::hal