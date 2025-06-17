# Petit

Petit provides optimized FP16/BF16 x FP4 GPU kernels specifically designed for AMD GPUs. It enables efficient execution of NVFP4 quantized models on GPUs that lack native FP4 arithmetic capabilities. This makes Petit particularly well-suited for serving high-quality NVFP4 models on standard GPUs while achieving ~3.3x memory savings. For example, a server with 8x AMD MI300 GPUs can serve the [Llama-3.3-70B-Instruct](meta-llama/Llama-3.3-70B-Instruct) / [Llama-3.3-70B-Instruct-FP4](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4) model with a MMLU score of 82.22 and 78.88 respectively.  

## Requirements

* AMD CDNA2 / CDNA3 GPUs (AMD MI2xx / MI3xx series)
* ROCm 6.2 or later
* PyTorch 2.5 or later

## Installation and Usages

You can install Petit directly using pip:

```bash
pip install .
```

You might need to specify `CMAKE_PREFIX_PATH` in the environment variables if pip fails to detect the ROCm or PyTorch.

Petit provides python APIs for matrix multiplications that are intended to be integrated with inference frameworks such as [SGLang](https://github.com/sgl-project/sglang) and [vLLM](https://github.com/vllm-project/vllm.git). It also provides C++ bindings to enable integrations with frameworks like [llama.cpp](https://github.com/ggml-org/llama.cpp.git). 

## Techniques and Evaluations

Petit adopts the core ideas from [Marlin](https://github.com/IST-DASLab/marlin.git) and tailors the ideas optimizations for the throughput-oriented CDNA2 and CDNA3 architectures. Detailed information about these optimizations is available in a separate article.

## Known Issues

Similar to Marlin, Petit shuffles the data offline to minimize the work performed on the GPU side. It requires all scales are positive which matches the output of the [ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer.git) quantizier. 

The MFMA instructions on AMD MI2xx GPUs flush input and output denormal values to zero, which can potentially impact [numeric accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices). Petit implements corrective measures for the AMD MI2xx GPUs which have ~10% overheads. 

Compared to NVIDIA architectures, CDNA architectures are significantly more sensitive to kernel hyperparameters, such as shared memory shapes. We strongly recommend running auto-tuning to achieve optimal performance. The repository provides benchmarking tool to facilitate auto tunings.

## Contacts and Contributions

We thank AMD for their generous support of providing cloud access of the GPUs to make this project possible. AMD is not involved in the development of the project.

Petit is a very young project and we are still working on implementing various optimizations.  Please contact haohui@causalflow.ai for questions and supports. Contributions are welcome.

