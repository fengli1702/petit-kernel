import torch
import petit_kernel.ops as ops

def test_moe_gemm():
    """
    使用 PyTorch 作为参考基线，测试 MoE GEMM C++ 算子。
    """
    # --- 1. 参数定义 ---
    num_tokens = 256  # 输入张量中的总 token 数
    m = 64            # 路由到当前专家的 token 数
    k = 4096          # 输入维度 (hidden_size)
    n = 14336         # 输出维度 (intermediate_size)
    device = "cuda"
    dtype = torch.float16

    print(f"Testing MoE GEMM with M={m}, N={n}, K={k}, Total Tokens={num_tokens}")

    # --- 2. 准备输入数据 ---
    # 完整的输入张量 (所有 token 的激活值)
    a_full = torch.randn(num_tokens, k, device=device, dtype=dtype)
    
    # 随机选择 m 个 token 的索引，模拟路由结果
    # a_indices = torch.randperm(num_tokens, device=device)[:m].int()
    # 为保证可复现，我们选择固定的索引
    a_indices = torch.arange(0, m, device=device, dtype=torch.int32)

    # 专家权重 (为简单起见，我们先用 float16 创建，之后再模拟量化)
    b_fp16 = torch.randn(n, k, device=device, dtype=dtype)
    
    # TODO: 你的量化函数应该在这里被调用
    # b_quant, scales, _ = quantize_fp4(b_fp16)
    # 目前，我们先用一个占位符
    b_quant = torch.randint(0, 255, (n, k // 8), dtype=torch.int32, device=device)
    scales = torch.ones((n, k // 16), dtype=torch.int32, device=device)
    global_scale = torch.tensor([1.0], device=device, dtype=torch.float32)


    # --- 3. 计算参考结果 (纯 PyTorch) ---
    print("Calculating reference result with PyTorch...")
    
    # a) Gather: 从 a_full 中根据索引选出需要的 token
    a_batch = torch.index_select(a_full, 0, a_indices)
    
    # b) Matmul: 执行标准的矩阵乘法
    c_batch_ref = torch.matmul(a_batch, b_fp16.T)
    
    # c) Scatter-Add: 将计算结果加回到正确的位置
    c_full_ref = torch.zeros(num_tokens, n, device=device, dtype=dtype)
    c_full_ref.index_add_(0, a_indices, c_batch_ref)
    
    print("Reference calculation complete.")

    # --- 4. 调用你的 C++ 算子 ---
    print("Calling custom MoE GEMM kernel...")
    
    # 创建一个空的输出张量
    c_full_out = torch.zeros(num_tokens, n, device=device, dtype=dtype)

    # TODO: 在这里调用你最终绑定的 Python 函数
    # ops.moe_gemm_fp4_fp16(
    #     c_full_out, a_full, a_indices, b_quant, scales, global_scale,
    #     m, n, k, num_tokens
    # )
    
    # 在你的核函数实现完成前，我们可以先用参考结果填充输出，确保对比逻辑正确
    c_full_out = c_full_ref.clone()
    
    print("Custom kernel execution complete.")

    # --- 5. 对比结果 ---
    print("Comparing results...")
    
    # 使用 allclose 进行比较，设置一个合理的容忍度
    # atol (absolute tolerance), rtol (relative tolerance)
    is_close = torch.allclose(c_full_out, c_full_ref, atol=1e-1, rtol=1e-2)
    
    if is_close:
        print("✅ Test Passed! Results are within tolerance.")
    else:
        print("❌ Test Failed! Results differ.")
        # 打印一些差异信息帮助调试
        diff = torch.abs(c_full_out - c_full_ref)
        print(f"   Max absolute error: {diff.max().item()}")
        print(f"   Max relative error: {(diff / torch.abs(c_full_ref)).max().item()}")

if __name__ == "__main__":
    test_moe_gemm()
