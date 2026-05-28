"""
PyTorch 设备管理测试 - 支持 Apple MPS / NVIDIA CUDA / CPU
参考: https://zh-v2.d2l.ai/chapter_deep-learning-computation/use-gpu.html
"""
import torch
from torch import nn


# ============================================================
# 1. 设备探测函数（适配 MPS / CUDA / CPU）
# ============================================================

def try_gpu(i=0):
    """返回可用的最佳设备: MPS > CUDA(i) > CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def try_all_gpus():
    """返回所有可用的 GPU 设备列表，都没有则返回 [cpu]"""
    devices = []
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    else:
        devices = [torch.device(f"cuda:{i}")
                   for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device("cpu")]


# ============================================================
# 2. 环境信息
# ============================================================

print("=" * 50)
print("PyTorch 版本:", torch.__version__)
print("=" * 50)

print(f"CUDA 可用:  {torch.cuda.is_available()}")
print(f"MPS 可用:   {torch.backends.mps.is_available()}")
print(f"MPS 已构建: {torch.backends.mps.is_built()}")

device = try_gpu()
print(f"\n使用设备:   {device}")
print(f"所有设备:   {try_all_gpus()}")
print("=" * 50)


# ============================================================
# 3. 张量与设备
# ============================================================

print("\n--- 张量设备测试 ---")

# 默认在 CPU
x = torch.tensor([1, 2, 3])
print(f"CPU 张量:      {x.device}")

# 在目标设备上创建
X = torch.ones(2, 3, device=device)
print(f"设备上张量:    {X.device}")

# CPU -> 设备
Y = torch.rand(2, 3).to(device)
print(f"to() 迁移后:   {Y.device}")

# 设备上计算
Z = X + Y
print(f"计算结果设备:  {Z.device}")
print(f"Z = {Z}")

# 拷回 CPU（用于 numpy 等）
Z_cpu = Z.cpu()
print(f"拷回 CPU:      {Z_cpu.device}")


# ============================================================
# 4. 模型与设备
# ============================================================

print("\n--- 模型设备测试 ---")

net = nn.Sequential(nn.Linear(3, 1))
print(f"模型参数初始设备: {list(net.parameters())[0].device}")

net = net.to(device)
print(f"模型迁移后设备:   {list(net.parameters())[0].device}")

# 前向推理（输入也必须在同一设备）
input_data = torch.rand(4, 3, device=device)
output = net(input_data)
print(f"输入设备:   {input_data.device}")
print(f"输出设备:   {output.device}")
print(f"输出值:\n{output}")


# ============================================================
# 5. 设备一致性检查
# ============================================================

print("\n--- 设备一致性 ---")
print(f"X 与 Y 同设备: {X.device == Y.device}")
print(f"X 与 x 同设备: {X.device == x.device}")


# ============================================================
# 6. MPS 特有测试（仅在 MPS 可用时执行）
# ============================================================

if device.type == "mps":
    print("\n--- MPS 特有测试 ---")

    # MPS 当前不支持 float64，确认 float32 正常
    a = torch.randn(1000, 1000, device="mps")
    b = torch.randn(1000, 1000, device="mps")
    c = a @ b  # 矩阵乘法
    print(f"大矩阵乘法结果设备: {c.device}")
    print(f"结果形状: {c.shape}")

    # autograd 测试
    x = torch.randn(3, requires_grad=True, device="mps")
    y = x.pow(2).sum()
    y.backward()
    print(f"梯度设备: {x.grad.device}")
    print(f"梯度值: {x.grad}")

print("\n全部测试通过!")
