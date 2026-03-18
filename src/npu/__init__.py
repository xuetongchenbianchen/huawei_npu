"""NPU 示例实现：优先使用 Ascend NPU（通过 torch + torch_npu），回退到 CPU 上的 torch。
接口同 CPU：init(), generate_input(), run_once()
"""
import warnings
import numpy as np


def _setup_torch():
    try:
        import torch
    except Exception:
        return None, None

    # 尝试使用 torch_npu（如果安装）
    device = torch.device('cpu')
    try:
        import torch_npu  # noqa: F401
        device = torch.device('npu:0')
    except Exception:
        # 如果没有 torch_npu，则使用 cpu（或普通 torch）
        warnings.warn('torch_npu 未找到，NPU 实现将回退到 CPU（若需 NPU，请安装 torch_npu）。')
    return torch, device


TORCH, DEVICE = _setup_torch()


def init():
    pass


def generate_input(shape=(512, 512), dtype='float32'):
    if TORCH is None:
        # 回退到 numpy-like lists
        import numpy as _np
        a = _np.random.randn(*shape).astype(_np.float32)
        b = _np.random.randn(shape[1], shape[1]).astype(_np.float32)
        return (a, b)

    a = TORCH.randn(*shape, dtype=getattr(TORCH, dtype)).to(DEVICE)
    b = TORCH.randn(shape[1], shape[1], dtype=getattr(TORCH, dtype)).to(DEVICE)
    return (a, b)


def run_once(inp):
    if TORCH is None:
        # 使用 numpy 计算以保证可运行
        a, b = inp
        return __import__('numpy').sum(a.dot(b))
    a, b = inp
    c = TORCH.matmul(a, b)
    # 强制同步：在 NPU 上可能需要调用 .cpu() 或 .numpy()，这里尽量保证与 device 同步
    if c.device.type == 'npu':
        # 若存在 to('cpu') 会触发数据拷贝并同步计算
        return c.to('cpu').numpy().sum()
    else:
        return c.sum().item()


def workload_size(inp):
    """与 CPU 一致的 FLOPs 估算函数。支持 torch.Tensor 或 numpy 数组输入。"""
    a, b = inp
    try:
        shape_a = a.shape
        shape_b = b.shape
    except Exception:
        # 非标准输入，返回 0
        return 0
    m, k = shape_a
    k2, n = shape_b
    if k != k2:
        return int(m * k + k2 * n)
    return int(2 * m * k * n)


# ---- 额外算子： elementwise add ----
def generate_input_add(shape=(1024, 1024), dtype='float32'):
    if TORCH is None:
        import numpy as _np
        a = _np.random.randn(*shape).astype(_np.float32)
        b = _np.random.randn(*shape).astype(_np.float32)
        return (a, b)
    a = TORCH.randn(*shape, dtype=getattr(TORCH, dtype)).to(DEVICE)
    b = TORCH.randn(*shape, dtype=getattr(TORCH, dtype)).to(DEVICE)
    return (a, b)


def run_once_add(inp):
    if TORCH is None:
        a, b = inp
        return __import__('numpy').add(a, b).sum()
    a, b = inp
    return TORCH.add(a, b).sum().to('cpu').item()


def workload_size_add(inp):
    a, _ = inp
    try:
        return int(a.numel())
    except Exception:
        return int(a.size)


# ---- 额外算子： conv2d using torch if available ----
def generate_input_conv2d(batch=1, in_c=3, h=32, w=32, out_c=8, k=3, dtype='float32'):
    if TORCH is None:
        import numpy as _np
        x = _np.random.randn(batch, in_c, h, w).astype(_np.float32)
        weight = _np.random.randn(out_c, in_c, k, k).astype(_np.float32)
        return (x, weight)
    x = TORCH.randn(batch, in_c, h, w, dtype=getattr(TORCH, dtype)).to(DEVICE)
    weight = TORCH.randn(out_c, in_c, k, k, dtype=getattr(TORCH, dtype)).to(DEVICE)
    return (x, weight)


def run_once_conv2d(inp):
    x, weight = inp
    if TORCH is None:
        # 调用 CPU 模块的朴素实现以复用代码
        cpu_mod = __import__('src.cpu', fromlist=['_conv2d_np'])
        return __import__('numpy').sum(cpu_mod._conv2d_np(x, weight))
    import torch.nn.functional as F
    out = F.conv2d(x, weight, bias=None, stride=1, padding=1)
    # 同步并返回标量
    return out.to('cpu').numpy().sum()


def workload_size_conv2d(inp):
    x, weight = inp
    try:
        batch, in_c, h, w = x.shape
        out_c, _, kH, kW = weight.shape
    except Exception:
        return 0
    out_h = h
    out_w = w
    return int(2 * out_c * out_h * out_w * in_c * kH * kW)


# ---- 额外算子： SDPA (scaled dot product attention) ----
def generate_input_sdpa(batch=2, heads=4, seq_len=128, head_dim=256, dtype='float32'):
    if TORCH is None:
        np_dtype = np.float32 if dtype == 'float32' else np.float16
        q = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
        k = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
        v = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
        return (q, k, v)

    q = TORCH.randn(batch, heads, seq_len, head_dim, dtype=getattr(TORCH, dtype)).to(DEVICE)
    k = TORCH.randn(batch, heads, seq_len, head_dim, dtype=getattr(TORCH, dtype)).to(DEVICE)
    v = TORCH.randn(batch, heads, seq_len, head_dim, dtype=getattr(TORCH, dtype)).to(DEVICE)
    return (q, k, v)


def _softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def run_once_sdpa(inp):
    q, k, v = inp
    if TORCH is not None:
        import torch.nn.functional as F
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.to('cpu').sum().item()

    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    attn = _softmax_np(scores, axis=-1)
    out = np.matmul(attn, v)
    return float(np.sum(out))


def workload_size_sdpa(inp):
    q, _, _ = inp
    b, h, s, d = q.shape
    return int(4 * b * h * s * s * d)
