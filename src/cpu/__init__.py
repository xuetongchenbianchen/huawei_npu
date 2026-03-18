"""CPU 示例实现：使用 NumPy 做矩阵乘法和简单后处理。
接口：
 - init() 可选
 - generate_input() -> 返回输入数据
 - run_once(input) -> 执行一次计算
"""
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def init():
    pass


def generate_input(shape=(512, 512), dtype=np.float32):
    a = np.random.randn(*shape).astype(dtype)
    b = np.random.randn(shape[1], shape[1]).astype(dtype)
    return (a, b)


def run_once(inp):
    a, b = inp
    # 矩阵乘法 + 求和（代表后处理）
    c = a.dot(b)
    return np.sum(c)


def workload_size(inp):
    """返回估算的浮点运算数量（FLOPs）。对于矩阵乘法 A(M,K) * B(K,N)，大约为 2*M*K*N FLOPs。"""
    a, b = inp
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        # 不能直接计算，返回元素数量作为保守估计
        return int(m * k + k2 * n)
    return int(2 * m * k * n)


# ---- 额外算子： elementwise add ----
def generate_input_add(shape=(1024, 1024), dtype=np.float32):
    a = np.random.randn(*shape).astype(dtype)
    b = np.random.randn(*shape).astype(dtype)
    return (a, b)


def run_once_add(inp):
    a, b = inp
    return np.add(a, b).sum()


def workload_size_add(inp):
    a, b = inp
    return int(a.size)  # 每个元素一次加法


# ---- 额外算子： 2D convolution (naive) ----
def generate_input_conv2d(batch=1, in_c=3, h=32, w=32, out_c=8, k=3, dtype=np.float32):
    x = np.random.randn(batch, in_c, h, w).astype(dtype)
    weight = np.random.randn(out_c, in_c, k, k).astype(dtype)
    return (x, weight)


def _conv2d_np(x, weight, stride=1, padding=1):
    batch, in_c, h, w = x.shape
    out_c, _, kH, kW = weight.shape
    out_h = (h + 2 * padding - kH) // stride + 1
    out_w = (w + 2 * padding - kW) // stride + 1
    out = np.zeros((batch, out_c, out_h, out_w), dtype=x.dtype)
    # pad
    xp = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    for b in range(batch):
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        h0 = i * stride
                        w0 = j * stride
                        out[b, oc, i, j] += np.sum(
                            xp[b, ic, h0:h0 + kH, w0:w0 + kW] * weight[oc, ic]
                        )
    return out


def run_once_conv2d(inp):
    x, weight = inp
    out = _conv2d_np(x, weight)
    return out.sum()


def workload_size_conv2d(inp):
    x, weight = inp
    batch, in_c, h, w = x.shape
    out_c, _, kH, kW = weight.shape
    out_h = h  # approximate with padding=stride=1
    out_w = w
    # 每个输出元素约需 2 * kH * kW * in_c FLOPs
    return int(2 * out_c * out_h * out_w * in_c * kH * kW)


# ---- 额外算子： SDPA (scaled dot product attention) ----
def generate_input_sdpa(batch=2, heads=4, seq_len=128, head_dim=256, dtype='float32'):
    if torch is not None:
        tdtype = getattr(torch, dtype)
        q = torch.randn(batch, heads, seq_len, head_dim, dtype=tdtype)
        k = torch.randn(batch, heads, seq_len, head_dim, dtype=tdtype)
        v = torch.randn(batch, heads, seq_len, head_dim, dtype=tdtype)
        return (q, k, v)

    np_dtype = np.float32 if dtype == 'float32' else np.float16
    q = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    k = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    v = np.random.randn(batch, heads, seq_len, head_dim).astype(np_dtype)
    return (q, k, v)


def _softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def run_once_sdpa(inp):
    q, k, v = inp

    if torch is not None and isinstance(q, torch.Tensor):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return out.sum().item()

    # numpy 回退实现
    scale = 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    attn = _softmax_np(scores, axis=-1)
    out = np.matmul(attn, v)
    return float(np.sum(out))


def workload_size_sdpa(inp):
    q, _, _ = inp
    shape = q.shape
    b, h, s, d = shape
    # 粗略估算：QK^T 与 Attn*V 各约 2*B*H*S*S*D
    return int(4 * b * h * s * s * d)
