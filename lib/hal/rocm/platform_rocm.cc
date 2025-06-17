#include "hal/device.h"
#include <absl/status/status.h>
#include <hip/hip_runtime.h>
#include <memory>

namespace causalflow::petit::hal {

namespace rocm {

class RocmPlatform : public Platform {
  public:
    virtual absl::Status GetDevice(int id,
                                   std::unique_ptr<Device> *result) override;
    virtual ~RocmPlatform() = default;
};

class RocmDevice : public Device {
  public:
    virtual absl::Status Malloc(void **ptr, size_t size) override {
        return ToStatus(hipMalloc(ptr, size));
    }

    virtual absl::Status Free(void *ptr) override {
        return ToStatus(hipFree(ptr));
    }

    virtual absl::Status Memset(void *ptr, int value, size_t size) override {
        return ToStatus(hipMemset(ptr, value, size));
    }

    virtual absl::Status CopyToDevice(void *dst, const void *src,
                                      size_t size) override {
        return ToStatus(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
    }

    virtual absl::Status CopyToHost(void *dst, const void *src,
                                    size_t size) override {
        return ToStatus(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
    }

    virtual absl::Status Synchronize() override {
        return ToStatus(hipDeviceSynchronize());
    }

    virtual ~RocmDevice() = default;

  private:
    absl::Status ToStatus(hipError_t err);
};

absl::Status RocmDevice::ToStatus(hipError_t err) {
    if (err != hipSuccess) {
        return absl::InternalError("Error");
    }
    return absl::OkStatus();
}

absl::Status RocmPlatform::GetDevice(int id, std::unique_ptr<Device> *result) {
    *result = std::unique_ptr<Device>(new RocmDevice);
    return absl::OkStatus();
}

} // namespace rocm

Platform *GetRocmPlatform() {
    static rocm::RocmPlatform inst;
    return &inst;
}

} // namespace causalflow::petit::hal