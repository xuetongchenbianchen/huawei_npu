"""CUDA benchmark backend.

This backend mirrors the src.cpu/src.npu function naming convention so the
benchmark runner can call the same generate_input_<op>, compute_<op>,
run_once_<op>, and workload_size_<op> functions for CUDA targets.

CUDA kernels are provided by PyTorch. If PyTorch or CUDA is unavailable, the
module falls back to CPU tensors or NumPy so imports and smoke checks still
work in CPU-only environments. Fallback results are not CUDA performance data.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def _setup_device():
    if torch is None:
        warnings.warn("PyTorch is not available; CUDA backend will use NumPy fallback.")
        return None, False
    try:
        if torch.cuda.is_available():
            return torch.device("cuda:0"), True
        warnings.warn("torch.cuda.is_available() is False; CUDA backend falls back to CPU.")
    except Exception as exc:
        warnings.warn(f"CUDA availability check failed; CUDA backend falls back to CPU: {exc}")
    return torch.device("cpu"), False


DEVICE, HAS_CUDA = _setup_device()


def init():
    if HAS_CUDA and torch is not None:
        torch.cuda.set_device(DEVICE)


def backend_info():
    info = {
        "torch_available": torch is not None,
        "cuda_available": bool(HAS_CUDA),
        "device": str(DEVICE) if DEVICE is not None else "numpy",
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "torch_version": None,
        "cuda_version": None,
    }
    if torch is not None:
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            if HAS_CUDA:
                info["cuda_device_name"] = torch.cuda.get_device_name(DEVICE)
        except Exception as exc:
            info["cuda_device_name"] = f"unavailable: {type(exc).__name__}: {exc}"
    return info


def synchronize():
    if HAS_CUDA and torch is not None:
        torch.cuda.synchronize(DEVICE)


def _resolve_np_dtype(dtype):
    if dtype in (None, "float32", np.float32):
        return np.float32
    if dtype in ("float16", np.float16):
        return np.float16
    if dtype in ("float64", np.float64):
        return np.float64
    if dtype in ("bool", bool, np.bool_):
        return np.bool_
    return np.float32


def _resolve_torch_dtype(dtype):
    if torch is None:
        return None
    if dtype in (None, "float32", np.float32):
        return torch.float32
    if dtype in ("float16", np.float16):
        return torch.float16
    if dtype in ("float64", np.float64):
        return torch.float64
    if dtype in ("int64", np.int64):
        return torch.int64
    if dtype in ("bool", bool, np.bool_):
        return torch.bool
    if dtype in ("bfloat16",):
        return getattr(torch, "bfloat16", torch.float32)
    return torch.float32


def _to_numpy(value):
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_device_tensor(value, dtype=None):
    if torch is None:
        return np.asarray(value, dtype=_resolve_np_dtype(dtype))
    target_dtype = dtype if dtype is not None and not isinstance(dtype, str) else _resolve_torch_dtype(dtype)
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            target_dtype = torch.bool
        elif not value.is_floating_point() and target_dtype in (None, torch.float32):
            target_dtype = value.dtype
        return value.to(device=DEVICE, dtype=target_dtype or value.dtype)
    if isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
        if value.dtype == np.bool_:
            target_dtype = torch.bool
        elif np.issubdtype(value.dtype, np.integer):
            target_dtype = torch.long
        elif target_dtype is None:
            target_dtype = _resolve_torch_dtype(value.dtype.type)
        return tensor.to(device=DEVICE, dtype=target_dtype)
    if isinstance(value, (bool, int, float)):
        return value
    return torch.tensor(value, dtype=target_dtype or torch.float32, device=DEVICE)


def materialize_input(inp):
    if isinstance(inp, tuple):
        return tuple(materialize_input(item) for item in inp)
    if isinstance(inp, list):
        return [materialize_input(item) for item in inp]
    if isinstance(inp, (bool, int, float)):
        return inp
    if torch is None:
        return np.asarray(inp)
    return _to_device_tensor(inp)


def _randn(*shape, dtype="float32"):
    if torch is None:
        return np.random.randn(*shape).astype(_resolve_np_dtype(dtype))
    return torch.randn(*shape, dtype=_resolve_torch_dtype(dtype), device=DEVICE)


def _randint(low, high, shape, dtype="int64"):
    if torch is None:
        return np.random.randint(low, high, size=shape, dtype=np.int64)
    return torch.randint(low, high, shape, dtype=_resolve_torch_dtype(dtype), device=DEVICE)


def _shape_product(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _numel(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return int(x.numel())
    return int(np.asarray(x).size)


def _scalar_output(output):
    if torch is not None and isinstance(output, torch.Tensor):
        return float(output.detach().sum().cpu().item())
    return float(np.asarray(output).sum())


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


# ---- Matmul ----
def generate_input(shape=(512, 512), dtype="float32"):
    return generate_input_matmul(shape=shape, dtype=dtype)


def generate_input_matmul(shape=(512, 512), dtype="float32"):
    a = _randn(*shape, dtype=dtype)
    b = _randn(shape[1], shape[1], dtype=dtype)
    return (a, b)


def compute_matmul(inp):
    a, b = inp
    if torch is not None:
        return torch.matmul(a, b)
    return np.matmul(_to_numpy(a), _to_numpy(b))


def run_once(inp):
    return run_once_matmul(inp)


def run_once_matmul(inp):
    return _scalar_output(compute_matmul(inp))


def workload_size(inp):
    return workload_size_matmul(inp)


def workload_size_matmul(inp):
    a, b = inp
    shape_a = tuple(int(x) for x in a.shape)
    shape_b = tuple(int(x) for x in b.shape)
    m, k = shape_a
    k2, n = shape_b
    if k != k2:
        return int(m * k + k2 * n)
    return int(2 * m * k * n)


# ---- Batch Matmul ----
def generate_input_bmm(batch=8, m=128, k=128, n=128, dtype="float32"):
    return (_randn(batch, m, k, dtype=dtype), _randn(batch, k, n, dtype=dtype))


def compute_bmm(inp):
    a, b = inp
    if torch is not None:
        return torch.bmm(a, b)
    return np.matmul(_to_numpy(a), _to_numpy(b))


def run_once_bmm(inp):
    return _scalar_output(compute_bmm(inp))


def workload_size_bmm(inp):
    a, b = inp
    batch, m, k = (int(v) for v in a.shape)
    _, k2, n = (int(v) for v in b.shape)
    if k != k2:
        return int(batch * (m * k + k2 * n))
    return int(2 * batch * m * k * n)


# ---- Elementwise binary ops ----
def _generate_input_binary(shape=(1024, 1024), dtype="float32", positive_second=False):
    a = _randn(*shape, dtype=dtype)
    b = _randn(*shape, dtype=dtype)
    if positive_second:
        if torch is not None and isinstance(b, torch.Tensor):
            b = torch.abs(b) + 1e-3
        else:
            b = np.abs(b) + np.asarray(1e-3, dtype=_resolve_np_dtype(dtype))
    return (a, b)


def _binary_workload_size(inp):
    a, _ = inp
    return _numel(a)


def generate_input_add(shape=(1024, 1024), dtype="float32"):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_add(inp):
    a, b = inp
    return torch.add(a, b) if torch is not None else np.add(_to_numpy(a), _to_numpy(b))


def run_once_add(inp):
    return _scalar_output(compute_add(inp))


def workload_size_add(inp):
    return _binary_workload_size(inp)


def generate_input_sub(shape=(1024, 1024), dtype="float32"):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_sub(inp):
    a, b = inp
    return torch.sub(a, b) if torch is not None else np.subtract(_to_numpy(a), _to_numpy(b))


def run_once_sub(inp):
    return _scalar_output(compute_sub(inp))


def workload_size_sub(inp):
    return _binary_workload_size(inp)


def generate_input_mul(shape=(1024, 1024), dtype="float32"):
    return _generate_input_binary(shape=shape, dtype=dtype)


def compute_mul(inp):
    a, b = inp
    return torch.mul(a, b) if torch is not None else np.multiply(_to_numpy(a), _to_numpy(b))


def run_once_mul(inp):
    return _scalar_output(compute_mul(inp))


def workload_size_mul(inp):
    return _binary_workload_size(inp)


def generate_input_div(shape=(1024, 1024), dtype="float32"):
    return _generate_input_binary(shape=shape, dtype=dtype, positive_second=True)


def compute_div(inp):
    a, b = inp
    return torch.div(a, b) if torch is not None else np.divide(_to_numpy(a), _to_numpy(b))


def run_once_div(inp):
    return _scalar_output(compute_div(inp))


def workload_size_div(inp):
    return _binary_workload_size(inp)


# ---- Activations ----
def generate_input_relu(shape=(1024, 1024), dtype="float32"):
    return (_randn(*shape, dtype=dtype),)


def compute_relu(inp):
    (x,) = inp
    return torch.relu(x) if torch is not None else np.maximum(_to_numpy(x), 0)


def run_once_relu(inp):
    return _scalar_output(compute_relu(inp))


def workload_size_relu(inp):
    (x,) = inp
    return _numel(x)


def generate_input_gelu(shape=(1024, 1024), dtype="float32"):
    return (_randn(*shape, dtype=dtype),)


def compute_gelu(inp):
    (x,) = inp
    if torch is not None and F is not None:
        return F.gelu(x)
    x_np = _to_numpy(x).astype(np.float32, copy=False)
    return 0.5 * x_np * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_np + 0.044715 * np.power(x_np, 3))))


def run_once_gelu(inp):
    return _scalar_output(compute_gelu(inp))


def workload_size_gelu(inp):
    (x,) = inp
    return int(8 * _numel(x))


def generate_input_silu(shape=(1024, 1024), dtype="float32"):
    return (_randn(*shape, dtype=dtype),)


def compute_silu(inp):
    (x,) = inp
    if torch is not None and F is not None:
        return F.silu(x)
    x_np = _to_numpy(x).astype(np.float32, copy=False)
    return x_np / (1.0 + np.exp(-x_np))


def run_once_silu(inp):
    return _scalar_output(compute_silu(inp))


def workload_size_silu(inp):
    (x,) = inp
    return int(5 * _numel(x))


# ---- Reductions ----
def generate_input_sum(shape=(1024, 1024), axis=1, dtype="float32"):
    return (_randn(*shape, dtype=dtype), int(axis))


def compute_sum(inp):
    x, axis = inp
    return torch.sum(x, dim=int(axis)) if torch is not None else np.sum(_to_numpy(x), axis=int(axis))


def run_once_sum(inp):
    return _scalar_output(compute_sum(inp))


def workload_size_sum(inp):
    x, _ = inp
    return _numel(x)


def generate_input_mean(shape=(1024, 1024), axis=1, dtype="float32"):
    return (_randn(*shape, dtype=dtype), int(axis))


def compute_mean(inp):
    x, axis = inp
    return torch.mean(x, dim=int(axis)) if torch is not None else np.mean(_to_numpy(x), axis=int(axis))


def run_once_mean(inp):
    return _scalar_output(compute_mean(inp))


def workload_size_mean(inp):
    x, _ = inp
    return _numel(x)


def generate_input_max(shape=(1024, 1024), axis=1, dtype="float32"):
    return (_randn(*shape, dtype=dtype), int(axis))


def compute_max(inp):
    x, axis = inp
    if torch is not None:
        return torch.max(x, dim=int(axis)).values
    return np.max(_to_numpy(x), axis=int(axis))


def run_once_max(inp):
    return _scalar_output(compute_max(inp))


def workload_size_max(inp):
    x, _ = inp
    return _numel(x)


# ---- Layout transform ----
def generate_input_transpose_contiguous(shape=(1024, 1024), dtype="float32"):
    return (_randn(*shape, dtype=dtype),)


def compute_transpose_contiguous(inp):
    (x,) = inp
    if torch is not None:
        return x.transpose(-1, -2).contiguous()
    return np.ascontiguousarray(np.swapaxes(_to_numpy(x), -1, -2))


def run_once_transpose_contiguous(inp):
    return _scalar_output(compute_transpose_contiguous(inp))


def workload_size_transpose_contiguous(inp):
    (x,) = inp
    return int(2 * _numel(x))


# ---- Conv2d ----
def generate_input_conv2d(
    batch=1,
    in_c=3,
    h=32,
    w=32,
    out_c=8,
    k=3,
    dtype="float32",
):
    x = _randn(batch, in_c, h, w, dtype=dtype)
    weight = _randn(out_c, in_c, k, k, dtype=dtype)
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
    if torch is not None and F is not None:
        return F.conv2d(x, weight, bias=None, stride=1, padding=1)
    return _conv2d_np(_to_numpy(x), _to_numpy(weight))


def run_once_conv2d(inp):
    return _scalar_output(compute_conv2d(inp))


def workload_size_conv2d(inp):
    x, weight = inp
    batch, in_c, h, w = (int(v) for v in x.shape)
    out_c, _, k_h, k_w = (int(v) for v in weight.shape)
    return int(2 * batch * out_c * h * w * in_c * k_h * k_w)


# ---- Pooling ----
def generate_input_maxpool2d(batch=4, channels=16, h=64, w=64, k=2, dtype="float32"):
    return (_randn(batch, channels, h, w, dtype=dtype), int(k))


def compute_maxpool2d(inp):
    x, k = inp
    if torch is not None and F is not None:
        return F.max_pool2d(x, kernel_size=int(k), stride=int(k))
    x_np = _to_numpy(x)
    batch, channels, h, w = x_np.shape
    out_h = h // int(k)
    out_w = w // int(k)
    x_view = x_np[:, :, : out_h * int(k), : out_w * int(k)].reshape(
        batch, channels, out_h, int(k), out_w, int(k)
    )
    return x_view.max(axis=(3, 5))


def run_once_maxpool2d(inp):
    return _scalar_output(compute_maxpool2d(inp))


def workload_size_maxpool2d(inp):
    x, k = inp
    return int(_numel(x) + _numel(x) // (int(k) * int(k)))


def generate_input_avgpool2d(batch=4, channels=16, h=64, w=64, k=2, dtype="float32"):
    return (_randn(batch, channels, h, w, dtype=dtype), int(k))


def compute_avgpool2d(inp):
    x, k = inp
    if torch is not None and F is not None:
        return F.avg_pool2d(x, kernel_size=int(k), stride=int(k))
    x_np = _to_numpy(x)
    batch, channels, h, w = x_np.shape
    out_h = h // int(k)
    out_w = w // int(k)
    x_view = x_np[:, :, : out_h * int(k), : out_w * int(k)].reshape(
        batch, channels, out_h, int(k), out_w, int(k)
    )
    return x_view.mean(axis=(3, 5))


def run_once_avgpool2d(inp):
    return _scalar_output(compute_avgpool2d(inp))


def workload_size_avgpool2d(inp):
    x, k = inp
    return int(_numel(x) + _numel(x) // (int(k) * int(k)))


# ---- Embedding and mask ----
def generate_input_embedding(vocab_size=8192, embedding_dim=256, batch=32, seq_len=128, dtype="float32"):
    weight = _randn(vocab_size, embedding_dim, dtype=dtype)
    indices = _randint(0, vocab_size, (batch, seq_len), dtype="int64")
    return (indices, weight)


def compute_embedding(inp):
    indices, weight = inp
    if torch is not None and F is not None:
        return F.embedding(indices.long(), weight)
    return _to_numpy(weight)[_to_numpy(indices).astype(np.int64, copy=False)]


def run_once_embedding(inp):
    return _scalar_output(compute_embedding(inp))


def workload_size_embedding(inp):
    indices, weight = inp
    return int(_numel(indices) * int(weight.shape[-1]))


def generate_input_where_mask(shape=(1024, 1024), true_ratio=0.5, dtype="float32"):
    if torch is not None:
        mask = torch.rand(*shape, device=DEVICE) < float(true_ratio)
    else:
        mask = np.random.random(shape) < float(true_ratio)
    return (mask, _randn(*shape, dtype=dtype), _randn(*shape, dtype=dtype))


def compute_where_mask(inp):
    mask, a, b = inp
    return torch.where(mask.bool(), a, b) if torch is not None else np.where(_to_numpy(mask).astype(bool), a, b)


def run_once_where_mask(inp):
    return _scalar_output(compute_where_mask(inp))


def workload_size_where_mask(inp):
    mask, _, _ = inp
    return int(3 * _numel(mask))


# ---- SDPA ----
def generate_input_sdpa(batch=2, heads=4, seq_len=128, head_dim=256, dtype="float32"):
    q = _randn(batch, heads, seq_len, head_dim, dtype=dtype)
    k = _randn(batch, heads, seq_len, head_dim, dtype=dtype)
    v = _randn(batch, heads, seq_len, head_dim, dtype=dtype)
    return (q, k, v)


def compute_sdpa(inp):
    q, k, v = inp
    if torch is not None and F is not None:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
    qn, kn, vn = (_to_numpy(x).astype(np.float32, copy=False) for x in inp)
    scale = 1.0 / math.sqrt(qn.shape[-1])
    scores = np.matmul(qn, np.swapaxes(kn, -2, -1)) * scale
    attn = _softmax_np(scores, axis=-1)
    return np.matmul(attn, vn)


def run_once_sdpa(inp):
    return _scalar_output(compute_sdpa(inp))


def workload_size_sdpa(inp):
    q = inp[0]
    b, h, s, d = (int(v) for v in q.shape)
    return int(4 * b * h * s * s * d)


# ---- Softmax ----
def generate_input_softmax(batch=32, features=4096, dtype="float32"):
    return (_randn(batch, features, dtype=dtype),)


def compute_softmax(inp):
    (x,) = inp
    return torch.softmax(x, dim=-1) if torch is not None else _softmax_np(_to_numpy(x).astype(np.float32), axis=-1)


def run_once_softmax(inp):
    return _scalar_output(compute_softmax(inp))


def workload_size_softmax(inp):
    (x,) = inp
    return int(5 * _numel(x))


# ---- LayerNorm ----
def generate_input_layernorm(batch=32, seq_len=128, hidden=1024, dtype="float32"):
    return (_randn(batch, seq_len, hidden, dtype=dtype),)


def compute_layernorm(inp, eps=1e-5):
    (x,) = inp
    if torch is not None and F is not None:
        return F.layer_norm(x, normalized_shape=(x.shape[-1],), weight=None, bias=None, eps=eps)
    return _layernorm_np(_to_numpy(x).astype(np.float32, copy=False), eps=eps)


def run_once_layernorm(inp):
    return _scalar_output(compute_layernorm(inp))


def workload_size_layernorm(inp):
    (x,) = inp
    return int(8 * _shape_product(x.shape))


# ---- RMSNorm ----
def generate_input_rmsnorm(batch=32, seq_len=128, hidden=1024, dtype="float32"):
    return (_randn(batch, seq_len, hidden, dtype=dtype),)


def compute_rmsnorm(inp, eps=1e-6):
    (x,) = inp
    if torch is not None:
        x_fp = x.float()
        rms = torch.sqrt(torch.mean(x_fp * x_fp, dim=-1, keepdim=True) + eps)
        return (x_fp / rms).to(dtype=x.dtype)
    return _rmsnorm_np(_to_numpy(x).astype(np.float32, copy=False), eps=eps)


def run_once_rmsnorm(inp):
    return _scalar_output(compute_rmsnorm(inp))


def workload_size_rmsnorm(inp):
    (x,) = inp
    return int(6 * _shape_product(x.shape))
