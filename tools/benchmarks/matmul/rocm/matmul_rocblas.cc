#include "matmul.h"
#include "utils/hip_helper.h"

#include <absl/status/status.h>
#include <absl/strings/escaping.h>
#include <hip/hip_fp16.h>
#include <rocblas/internal/rocblas-beta.h>
#include <rocblas/internal/rocblas-types.h>
#include <rocblas/rocblas.h>

namespace causalflow::petit::benchmark::matmul {

namespace rocm {

static inline void CheckRocblasStatus(rocblas_status status) {
    if (status != rocblas_status_success) {
        std::cerr << "RocBLAS Error: " << status << std::endl;
        throw std::runtime_error("RocBLAS Error");
    }
}

static constexpr float kAlpha = 1.0f;
static constexpr float kBeta = 0.0f;

class RocmStridedBatchMatmul : public Matmul {
  public:
    static constexpr size_t kWorkspaceSize = 32 * 1024 * 1024;
    explicit RocmStridedBatchMatmul(int m, int n, int k,
                                    const Matmul::DataType a_type,
                                    const Matmul::DataType c_type);

    absl::Status PrepareForBatchExecution(void *d, const void *a, const void *b,
                                          const void *c, long stride_a,
                                          long stride_b, long stride_c,
                                          int batch) override;
    absl::Status SetAlgorithm(AlgorithmDescriptor algo) override;
    absl::Status EnumerateAlgorithms() override;
    virtual size_t GetAlgorithmCount() const override {
        return algorithms_.size();
    }
    virtual std::string GetAlgorithmRepr(size_t index) const override {
        return std::string(reinterpret_cast<const char *>(&algorithms_[index]),
                           sizeof(int));
    }
    absl::Status Execute(size_t repeat) override;
    virtual ~RocmStridedBatchMatmul();

  private:
    static rocblas_datatype GetRocblasDataType(const Matmul::DataType type);
    rocblas_status GetSolutionsHelper(int *solution_list, int *size);
    absl::Status Close();

    rocblas_handle handle_;
    void *d_workspace_;
    void *d_d_;
    const void *d_a_, *d_b_, *d_c_;
    int m_, n_, k_, batch_;
    long stride_a_, stride_b_, stride_c_;
    rocblas_datatype element_type_, result_type_;

    rocblas_gemm_algo gemm_algo_;
    int solution_index_;
    std::vector<int> algorithms_;
};

class RocmMatmulFactory : public MatmulFactory {
  public:
    virtual const char *GetPlatformName() const override { return "rocm"; }
    virtual absl::Status
    CreateMatmul(hal::Device *dev, const Matmul::DataType a_type,
                 const Matmul::DataType c_type, int m, int n, int k,
                 std::unique_ptr<Matmul> *result) override {
        *result =
            std::make_unique<RocmStridedBatchMatmul>(m, n, k, a_type, c_type);
        return absl::OkStatus();
    }

    virtual ~RocmMatmulFactory() = default;
};

rocblas_datatype
RocmStridedBatchMatmul::GetRocblasDataType(const Matmul::DataType type) {
    switch (type) {
    case Matmul::DataType::kFp16:
        return rocblas_datatype_f16_r;
    case Matmul::DataType::kFp32:
        return rocblas_datatype_f32_r;
    case Matmul::DataType::kBf16:
        return rocblas_datatype_bf16_r;
    case Matmul::DataType::kFp8e4m3:
        return rocblas_datatype_f8_r;
    case Matmul::DataType::kFp8e5m2:
        return rocblas_datatype_bf8_r;
    }
    throw std::runtime_error("Invalid data type");
}

RocmStridedBatchMatmul::RocmStridedBatchMatmul(int m, int n, int k,
                                               const Matmul::DataType a_type,
                                               const Matmul::DataType c_type)
    : m_(m), n_(n), k_(k), gemm_algo_(rocblas_gemm_algo_standard),
      solution_index_(0) {
    element_type_ = GetRocblasDataType(a_type);
    result_type_ = GetRocblasDataType(c_type);

    CheckRocblasStatus(rocblas_create_handle(&handle_));
    CheckHIPStatus(hipMalloc(&d_workspace_, kWorkspaceSize));
}

RocmStridedBatchMatmul::~RocmStridedBatchMatmul() {
    auto stat = Close();
    (void)stat;
}

absl::Status RocmStridedBatchMatmul::Close() {
    CheckRocblasStatus(rocblas_destroy_handle(handle_));
    CheckHIPStatus(hipFree(d_workspace_));
    return absl::OkStatus();
}

absl::Status RocmStridedBatchMatmul::PrepareForBatchExecution(
    void *d, const void *a, const void *b, const void *c, long stride_a,
    long stride_b, long stride_c, int batch) {
    d_d_ = d;
    d_a_ = a;
    d_b_ = b;
    d_c_ = c;
    batch_ = batch;
    stride_a_ = stride_a;
    stride_b_ = stride_b;
    stride_c_ = stride_c;

    return absl::OkStatus();
}

absl::Status RocmStridedBatchMatmul::SetAlgorithm(AlgorithmDescriptor algo) {
    if (algo.tag == AlgorithmDescriptor::kDefault) {
        gemm_algo_ = rocblas_gemm_algo_standard;
        solution_index_ = 0;
    } else if (algo.tag == AlgorithmDescriptor::kIndex) {
        gemm_algo_ = rocblas_gemm_algo_solution_index;
        solution_index_ = algorithms_[std::stoi(algo.repr)];
    } else if (algo.tag == AlgorithmDescriptor::kOpaqueRepresentation) {
        gemm_algo_ = rocblas_gemm_algo_solution_index;
        solution_index_ = *reinterpret_cast<const int *>(algo.repr.c_str());
    }
    return absl::OkStatus();
}

rocblas_status RocmStridedBatchMatmul::GetSolutionsHelper(int *solution_list,
                                                          int *size) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    return rocblas_gemm_strided_batched_ex_get_solutions(
        handle_, rocblas_operation_transpose, rocblas_operation_none, m_, n_,
        k_, &kAlpha, d_a_, element_type_, k_, stride_a_, d_b_, element_type_,
        k_, stride_b_, &kBeta, d_c_, result_type_, m_, stride_c_, d_d_,
        result_type_, m_, stride_c_, batch_, rocblas_datatype_f32_r,
        rocblas_gemm_algo_solution_index, rocblas_gemm_flags_none,
        solution_list, size);
#pragma GCC diagnostic pop
}

absl::Status RocmStridedBatchMatmul::EnumerateAlgorithms() {
    int n_solutions;
    CheckRocblasStatus(GetSolutionsHelper(nullptr, &n_solutions));
    algorithms_.resize(n_solutions);
    CheckRocblasStatus(GetSolutionsHelper(algorithms_.data(), &n_solutions));

    return absl::OkStatus();
}

absl::Status RocmStridedBatchMatmul::Execute(size_t repeat) {
    for (size_t i = 0; i < repeat; ++i) {
        // Let A be in row-major and B in col-major to keep consistency with
        // densse-linears.
        CheckRocblasStatus(rocblas_gemm_strided_batched_ex(
            handle_, rocblas_operation_transpose, rocblas_operation_none, m_,
            n_, k_, &kAlpha, d_a_, element_type_, k_, stride_a_, d_b_,
            element_type_, k_, stride_b_, &kBeta, d_c_, result_type_, m_,
            stride_c_, d_d_, result_type_, m_, stride_c_, batch_,
            rocblas_datatype_f32_r, gemm_algo_, solution_index_, 0));
    }

    return absl::OkStatus();
}

} // namespace rocm

std::unique_ptr<MatmulFactory> CreateMatmulFactoryRocBLASBackend() {
    return std::unique_ptr<MatmulFactory>(new rocm::RocmMatmulFactory);
}

} // namespace causalflow::petit::benchmark::matmul
