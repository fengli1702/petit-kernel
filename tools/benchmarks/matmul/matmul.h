#include "hal/device.h"

#include <absl/status/status.h>
#include <memory>

namespace causalflow::petit::benchmark::matmul {

class Matmul {
  public:
    enum DataType {
        kFp8e5m2,
        kFp8e4m3,
        kFp16,
        kBf16,
        kFp32,
    };
    // Describe an algorithm used in the GEMM operation. Note that the indicies
    // are potentially non-deterministic.
    struct AlgorithmDescriptor {
        enum { kDefault, kIndex, kOpaqueRepresentation } tag;
        std::string repr;
    };
    // Stride batch GEMM
    virtual absl::Status PrepareForBatchExecution(void *d, const void *a,
                                                  const void *b, const void *c,
                                                  long stride_a, long stride_b,
                                                  long stride_c,
                                                  int batch_count) = 0;

    // Enumerate algorithms for tuning
    virtual absl::Status EnumerateAlgorithms() { return absl::OkStatus(); }
    virtual size_t GetAlgorithmCount() const { return 0; }
    virtual std::string GetAlgorithmRepr(size_t index) const { return ""; }

    virtual absl::Status SetAlgorithm(AlgorithmDescriptor algo) = 0;
    virtual absl::Status Execute(size_t repeat) = 0;
    virtual ~Matmul() = default;
};

class MatmulFactory {
  public:
    static std::unique_ptr<MatmulFactory> Create(const std::string &backend);
    virtual const char *GetPlatformName() const = 0;
    virtual absl::Status CreateMatmul(hal::Device *dev, Matmul::DataType a_type,
                                      Matmul::DataType c_type, int m, int n,
                                      int k,
                                      std::unique_ptr<Matmul> *result) = 0;
    virtual ~MatmulFactory() = default;
};

} // namespace causalflow::petit::benchmark::matmul