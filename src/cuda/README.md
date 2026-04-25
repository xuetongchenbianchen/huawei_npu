# CUDA Backend

`src.cuda` provides a PyTorch CUDA backend with the same public interface used by
`src.cpu` and `src.npu`.

Supported backend-level functions:

- `init()`
- `backend_info()`
- `synchronize()`
- `materialize_input(inp)`

Supported operator function groups:

- `generate_input_<op>(...)`
- `compute_<op>(inp)`
- `run_once_<op>(inp)`
- `workload_size_<op>(inp)`

Implemented operators:

- `matmul`
- `bmm`
- `add`
- `sub`
- `mul`
- `div`
- `relu`
- `gelu`
- `silu`
- `sum`
- `mean`
- `max`
- `transpose_contiguous`
- `conv2d`
- `maxpool2d`
- `avgpool2d`
- `embedding`
- `where_mask`
- `sdpa`
- `softmax`
- `layernorm`
- `rmsnorm`

The backend uses `cuda:0` when `torch.cuda.is_available()` is true. If PyTorch
or CUDA is unavailable, it falls back to CPU tensors or NumPy so the module can
still be imported in CPU-only development environments. Fallback execution is
only for interface checks and must not be reported as CUDA GPU performance.

Expected future benchmark usage after the device CLI work is merged:

```bash
python run_benchmark.py \
  --reference-device cpu \
  --target-device cuda \
  --ops matmul,conv2d,softmax \
  --scan \
  --check-precision
```

Current module-style usage:

```bash
python run_benchmark.py \
  --cpu-module src.cpu \
  --npu-module src.cuda \
  --ops matmul,add,relu \
  --repeat 5 \
  --warmup 2 \
  --check-precision
```

