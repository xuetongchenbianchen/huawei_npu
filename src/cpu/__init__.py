"""CPU benchmark backend.

提供基础算子与常见 NPU 向量算子的 CPU 参考实现，接口约定：
- init()
- generate_input_<op>()
- run_once_<op>(inp)
- compute_<op>(inp): 返回完整输出，供精度校验
- workload_size_<op>(inp)
"""
from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


DEFAULT_DTYPE = np.float32


def init():
    pass


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_np_dtype(dtype):
    if dtype in (None, "float32", np.float32):
        return np.float32
    if dtype in ("float16", np.float16):
        return np.float16
    if dtype in ("float64", np.float64):
        return np.float64
    return np.float32


def _randn(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)


def _softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _layernorm_np(x, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def _rmsnorm_np(x, eps=1e-6):
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps)
    return x / rms


def _value_sum(output):
    arr = _to_numpy(output)
    return float(arr.sum())


def _shape_product(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


# ---- Matmul ----
def generate_input(shape=(512, 512), dtype=np.float32):
    a = _randn(shape, _resolve_np_dtype(dtype))
    b = _randn((shape[1], shape[1]), _resolve_np_dtype(dtype))
    return (a, b)


def generate_input_matmul(shape=(512, 512), dtype=np.float32):
    return generate_input(shape=shape, dtype=dtype)


def compute_matmul(inp):
    a, b = inp
    return _to_numpy(a).dot(_to_numpy(b))


def run_once(inp):
    return run_once_matmul(inp)


def run_once_matmul(inp):
    return _value_sum(compute_matmul(inp))


def workload_size(inp):
    return workload_size_matmul(inp)


def workload_size_matmul(inp):
    a, b = inp
    a = _to_numpy(a)
    b = _to_numpy(b)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        return int(m * k + k2 * n)
    return int(2 * m * k * n)


# ---- Batch Matmul ----
def generate_input_bmm(batch=8, m=128, k=128, n=128, dtype=np.float32):
    np_dtype = _resolve_np_dtype(dtype)
    return (_randn((batch, m, k), np_dtype), _randn((batch, k, n), np_dtype))


def compute_bmm(inp):
    a, b = inp
    return np.matmul(_to_numpy(a), _to_numpy(b))


def run_once_bmm(inp):
    return _value_sum(compute_bmm(inp))


def workload_size_bmm(inp):
    a, b = inp
    batch, m, k = _to_numpy(a).shape
    _, k2, n = _to_numpy(b).shape
    if k != k2:
        return int(batch * (m * k + k2 * n))
    return int(2 * batch * m * k * n)


# ---- Elementwise binary ops ----
def _generate_input_binary(shape=(1024, 1024), dtype=np.float32, positive_second=False):
    np_dtype = _resolve_np_dtype(dtype)
    a = _randn(shape, np_dtype)
    b = _randn(shape, np_dtype)
    if positive_second:
        b = np.abs(b) + np.asarray(1e-3, dtype=np_dtype)
    return (a, b)


def _binary_workload_size(inp):
    a, _ = inp
    return int(_to_numpy(a).size)


def generate_input_add(shape=(1024, 1024), dtype=np.float32):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_add(inp):
    a, b = inp
    return np.add(_to_numpy(a), _to_numpy(b))


def run_once_add(inp):
    return _value_sum(compute_add(inp))


def workload_size_add(inp):
    return _binary_workload_size(inp)


def generate_input_sub(shape=(1024, 1024), dtype=np.float32):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_sub(inp):
    a, b = inp
    return np.subtract(_to_numpy(a), _to_numpy(b))


def run_once_sub(inp):
    return _value_sum(compute_sub(inp))


def workload_size_sub(inp):
    return _binary_workload_size(inp)


def generate_input_mul(shape=(1024, 1024), dtype=np.float32):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_mul(inp):
    a, b = inp
    return np.multiply(_to_numpy(a), _to_numpy(b))


def run_once_mul(inp):
    return _value_sum(compute_mul(inp))


def workload_size_mul(inp):
    return _binary_workload_size(inp)


def generate_input_div(shape=(1024, 1024), dtype=np.float32):
    return _generate_input_binary(shape=shape, dtype=dtype, positive_second=True)


def compute_div(inp):
    a, b = inp
    return np.divide(_to_numpy(a), _to_numpy(b))


def run_once_div(inp):
    return _value_sum(compute_div(inp))


def workload_size_div(inp):
    return _binary_workload_size(inp)


# ---- Activations ----
def generate_input_relu(shape=(1024, 1024), dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)),)


def compute_relu(inp):
    (x,) = inp
    return np.maximum(_to_numpy(x), 0)


def run_once_relu(inp):
    return _value_sum(compute_relu(inp))


def workload_size_relu(inp):
    (x,) = inp
    return int(_to_numpy(x).size)


def generate_input_gelu(shape=(1024, 1024), dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)),)


def compute_gelu(inp):
    (x,) = inp
    x = _to_numpy(x).astype(np.float32, copy=False)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def run_once_gelu(inp):
    return _value_sum(compute_gelu(inp))


def workload_size_gelu(inp):
    (x,) = inp
    return int(8 * _to_numpy(x).size)


def generate_input_silu(shape=(1024, 1024), dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)),)


def compute_silu(inp):
    (x,) = inp
    x = _to_numpy(x).astype(np.float32, copy=False)
    return x / (1.0 + np.exp(-x))


def run_once_silu(inp):
    return _value_sum(compute_silu(inp))


def workload_size_silu(inp):
    (x,) = inp
    return int(5 * _to_numpy(x).size)


# ---- Reductions ----
def generate_input_sum(shape=(1024, 1024), axis=1, dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)), int(axis))


def compute_sum(inp):
    x, axis = inp
    return np.sum(_to_numpy(x), axis=int(axis))


def run_once_sum(inp):
    return _value_sum(compute_sum(inp))


def workload_size_sum(inp):
    x, _ = inp
    return int(_to_numpy(x).size)


def generate_input_mean(shape=(1024, 1024), axis=1, dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)), int(axis))


def compute_mean(inp):
    x, axis = inp
    return np.mean(_to_numpy(x), axis=int(axis))


def run_once_mean(inp):
    return _value_sum(compute_mean(inp))


def workload_size_mean(inp):
    x, _ = inp
    return int(_to_numpy(x).size)


def generate_input_max(shape=(1024, 1024), axis=1, dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)), int(axis))


def compute_max(inp):
    x, axis = inp
    return np.max(_to_numpy(x), axis=int(axis))


def run_once_max(inp):
    return _value_sum(compute_max(inp))


def workload_size_max(inp):
    x, _ = inp
    return int(_to_numpy(x).size)


# ---- Layout transform ----
def generate_input_transpose_contiguous(shape=(1024, 1024), dtype=np.float32):
    return (_randn(shape, _resolve_np_dtype(dtype)),)


def compute_transpose_contiguous(inp):
    (x,) = inp
    return np.ascontiguousarray(np.swapaxes(_to_numpy(x), -1, -2))


def run_once_transpose_contiguous(inp):
    return _value_sum(compute_transpose_contiguous(inp))


def workload_size_transpose_contiguous(inp):
    (x,) = inp
    return int(2 * _to_numpy(x).size)


# ---- Conv2d ----
def generate_input_conv2d(
    batch=1,
    in_c=3,
    h=32,
    w=32,
    out_c=8,
    k=3,
    dtype=np.float32,
):
    np_dtype = _resolve_np_dtype(dtype)
    x = _randn((batch, in_c, h, w), np_dtype)
    weight = _randn((out_c, in_c, k, k), np_dtype)
    return (x, weight)


def _conv2d_np(x, weight, stride=1, padding=1):
    batch, in_c, h, w = x.shape
    out_c, _, k_h, k_w = weight.shape
    out_h = (h + 2 * padding - k_h) // stride + 1
    out_w = (w + 2 * padding - k_w) // stride + 1
    out = np.zeros((batch, out_c, out_h, out_w), dtype=x.dtype)
    x_pad = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    for b in range(batch):
        for oc in range(out_c):
            for ic in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        h0 = i * stride
                        w0 = j * stride
                        out[b, oc, i, j] += np.sum(
                            x_pad[b, ic, h0 : h0 + k_h, w0 : w0 + k_w] * weight[oc, ic]
                        )
    return out


def compute_conv2d(inp):
    x, weight = inp
    x = _to_numpy(x)
    weight = _to_numpy(weight)

    if torch is not None:
        xt = torch.from_numpy(x)
        wt = torch.from_numpy(weight)
        out = F.conv2d(xt, wt, bias=None, stride=1, padding=1)
        return out.detach().cpu().numpy()

    return _conv2d_np(x, weight)


def run_once_conv2d(inp):
    return _value_sum(compute_conv2d(inp))


def workload_size_conv2d(inp):
    x, weight = inp
    x = _to_numpy(x)
    weight = _to_numpy(weight)
    _, in_c, h, w = x.shape
    out_c, _, k_h, k_w = weight.shape
    return int(2 * out_c * h * w * in_c * k_h * k_w)


# ---- Pooling ----
def generate_input_maxpool2d(batch=4, channels=16, h=64, w=64, k=2, dtype=np.float32):
    return (_randn((batch, channels, h, w), _resolve_np_dtype(dtype)), int(k))


def compute_maxpool2d(inp):
    x, k = inp
    x = _to_numpy(x)
    if torch is not None and F is not None:
        xt = torch.from_numpy(x)
        return F.max_pool2d(xt, kernel_size=int(k), stride=int(k)).detach().cpu().numpy()
    batch, channels, h, w = x.shape
    out_h = h // int(k)
    out_w = w // int(k)
    x_view = x[:, :, : out_h * int(k), : out_w * int(k)].reshape(batch, channels, out_h, int(k), out_w, int(k))
    return x_view.max(axis=(3, 5))


def run_once_maxpool2d(inp):
    return _value_sum(compute_maxpool2d(inp))


def workload_size_maxpool2d(inp):
    x, k = inp
    return int(_to_numpy(x).size + _to_numpy(x).size // (int(k) * int(k)))


def generate_input_avgpool2d(batch=4, channels=16, h=64, w=64, k=2, dtype=np.float32):
    return (_randn((batch, channels, h, w), _resolve_np_dtype(dtype)), int(k))


def compute_avgpool2d(inp):
    x, k = inp
    x = _to_numpy(x)
    if torch is not None and F is not None:
        xt = torch.from_numpy(x)
        return F.avg_pool2d(xt, kernel_size=int(k), stride=int(k)).detach().cpu().numpy()
    batch, channels, h, w = x.shape
    out_h = h // int(k)
    out_w = w // int(k)
    x_view = x[:, :, : out_h * int(k), : out_w * int(k)].reshape(batch, channels, out_h, int(k), out_w, int(k))
    return x_view.mean(axis=(3, 5))


def run_once_avgpool2d(inp):
    return _value_sum(compute_avgpool2d(inp))


def workload_size_avgpool2d(inp):
    x, k = inp
    return int(_to_numpy(x).size + _to_numpy(x).size // (int(k) * int(k)))


# ---- Embedding and mask ----
def generate_input_embedding(vocab_size=8192, embedding_dim=256, batch=32, seq_len=128, dtype=np.float32):
    weight = _randn((vocab_size, embedding_dim), _resolve_np_dtype(dtype))
    indices = np.random.randint(0, vocab_size, size=(batch, seq_len), dtype=np.int64)
    return (indices, weight)


def compute_embedding(inp):
    indices, weight = inp
    return _to_numpy(weight)[_to_numpy(indices).astype(np.int64, copy=False)]


def run_once_embedding(inp):
    return _value_sum(compute_embedding(inp))


def workload_size_embedding(inp):
    indices, weight = inp
    return int(_to_numpy(indices).size * _to_numpy(weight).shape[-1])


def generate_input_where_mask(shape=(1024, 1024), true_ratio=0.5, dtype=np.float32):
    np_dtype = _resolve_np_dtype(dtype)
    mask = np.random.random(shape) < float(true_ratio)
    return (mask, _randn(shape, np_dtype), _randn(shape, np_dtype))


def compute_where_mask(inp):
    mask, a, b = inp
    return np.where(_to_numpy(mask).astype(bool, copy=False), _to_numpy(a), _to_numpy(b))


def run_once_where_mask(inp):
    return _value_sum(compute_where_mask(inp))


def workload_size_where_mask(inp):
    mask, _, _ = inp
    return int(3 * _to_numpy(mask).size)


# ---- SDPA ----
def generate_input_sdpa(batch=2, heads=4, seq_len=128, head_dim=256, dtype="float32"):
    np_dtype = _resolve_np_dtype(dtype)
    q = _randn((batch, heads, seq_len, head_dim), np_dtype)
    k = _randn((batch, heads, seq_len, head_dim), np_dtype)
    v = _randn((batch, heads, seq_len, head_dim), np_dtype)
    return (q, k, v)


def compute_sdpa(inp):
    q, k, v = (_to_numpy(x).astype(np.float32, copy=False) for x in inp)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    attn = _softmax_np(scores, axis=-1)
    return np.matmul(attn, v)


def run_once_sdpa(inp):
    if torch is not None:
        q, k, v = inp
        q = torch.from_numpy(_to_numpy(q))
        k = torch.from_numpy(_to_numpy(k))
        v = torch.from_numpy(_to_numpy(v))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        return float(out.detach().cpu().numpy().sum())
    return _value_sum(compute_sdpa(inp))


def workload_size_sdpa(inp):
    q = _to_numpy(inp[0])
    b, h, s, d = q.shape
    return int(4 * b * h * s * s * d)


# ---- Softmax ----
def generate_input_softmax(batch=32, features=4096, dtype=np.float32):
    return (_randn((batch, features), _resolve_np_dtype(dtype)),)


def compute_softmax(inp):
    (x,) = inp
    x = _to_numpy(x).astype(np.float32, copy=False)
    return _softmax_np(x, axis=-1)


def run_once_softmax(inp):
    return _value_sum(compute_softmax(inp))


def workload_size_softmax(inp):
    (x,) = inp
    x = _to_numpy(x)
    return int(5 * x.size)


# ---- LayerNorm ----
def generate_input_layernorm(batch=32, seq_len=128, hidden=1024, dtype=np.float32):
    return (_randn((batch, seq_len, hidden), _resolve_np_dtype(dtype)),)


def compute_layernorm(inp):
    (x,) = inp
    return _layernorm_np(_to_numpy(x).astype(np.float32, copy=False))


def run_once_layernorm(inp):
    return _value_sum(compute_layernorm(inp))


def workload_size_layernorm(inp):
    (x,) = inp
    return int(8 * _shape_product(_to_numpy(x).shape))


# ---- RMSNorm ----
def generate_input_rmsnorm(batch=32, seq_len=128, hidden=1024, dtype=np.float32):
    return (_randn((batch, seq_len, hidden), _resolve_np_dtype(dtype)),)


def compute_rmsnorm(inp):
    (x,) = inp
    return _rmsnorm_np(_to_numpy(x).astype(np.float32, copy=False))


def run_once_rmsnorm(inp):
    return _value_sum(compute_rmsnorm(inp))


def workload_size_rmsnorm(inp):
    (x,) = inp
    return int(6 * _shape_product(_to_numpy(x).shape))
