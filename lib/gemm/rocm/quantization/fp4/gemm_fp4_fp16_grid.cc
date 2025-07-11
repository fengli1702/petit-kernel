#include "gemm/rocm/quantization/gemm.h"
#include "gemm_fp4.h"

namespace causalflow::petit::rocm::quantization::fp4 {

unsigned long ChooseDefaultFp4Fp16Solution(unsigned m, unsigned n, unsigned k,
                                           const PetitSolutionHints &hints);

class Dispatcher {
  public:
    using Call = SolutionMap::Call;

    static Dispatcher &GetInstance() {
        static Dispatcher instance;
        return instance;
    }

    int Dispatch(unsigned *c, const unsigned *a, const unsigned *b,
                 const unsigned *scales, const float *global_scale,
                 const unsigned m, const unsigned n, const unsigned k,
                 unsigned long solution_id, hipStream_t stream) {
        const auto it = solution_id_to_call_.find(solution_id);
        if (it == solution_id_to_call_.end()) {
            return kErrorKernelShape;
        }
        return it->second(c, a, b, scales, global_scale, m, n, k, stream);
    }

  protected:
    std::unordered_map<unsigned long, Call> solution_id_to_call_;
    Dispatcher() : solution_id_to_call_(SolutionMap::GetDispatchEntries()) {}
};

int GemmFp4Fp16Grid(unsigned *c, const unsigned *a, const unsigned *b,
                    const unsigned *scales, const float *global_scale,
                    const unsigned m, const unsigned n, const unsigned k,
                    const PetitSolutionHints &hints, unsigned long solution_id,
                    hipStream_t stream) {
    if (m == 0 || n == 0 || k == 0) {
        return 0;
    }

    if (solution_id == -1) {
        solution_id = ChooseDefaultFp4Fp16Solution(m, n, k, hints);
    }

    if (solution_id == -1) {
        return kErrorProblemShape;
    }

    return Dispatcher::GetInstance().Dispatch(c, a, b, scales, global_scale, m,
                                              n, k, solution_id, stream);
}

} // namespace causalflow::petit::rocm::quantization::fp4