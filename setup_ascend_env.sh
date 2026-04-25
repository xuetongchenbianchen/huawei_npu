#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASCEND_HOME_DEFAULT="/usr/local/Ascend/ascend-toolkit/latest"
ASCEND_HOME="${ASCEND_HOME:-$ASCEND_HOME_DEFAULT}"

echo "[1/4] 检查 uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装 uv。"
  exit 1
fi

echo "[2/4] 加载 Ascend 环境"
if [ -f "$ASCEND_HOME/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "$ASCEND_HOME/set_env.sh"
  echo "已加载: $ASCEND_HOME/set_env.sh"
else
  echo "未找到 $ASCEND_HOME/set_env.sh"
  echo "请确认 CANN/Ascend Toolkit 已安装，并设置 ASCEND_HOME。"
  exit 1
fi

echo "[3/4] 同步 Python 依赖"
cd "$ROOT_DIR"
uv sync --extra ascend
# torch_npu 2.1.0 仍依赖 pkg_resources，需使用兼容版 setuptools
uv pip install "setuptools<81" >/dev/null

echo "[4/4] 校验 torch_npu"
uv run python - <<'PY'
import sys

try:
    import torch
    import torch_npu  # noqa: F401
except Exception as exc:
    print("导入失败:", exc)
    sys.exit(1)

print("torch version:", torch.__version__)
if hasattr(torch, "npu"):
    try:
        print("torch.npu.is_available():", torch.npu.is_available())
        if torch.npu.is_available():
            print("torch.npu.device_count():", torch.npu.device_count())
    except Exception as exc:
        print("NPU 状态检查失败:", exc)
        sys.exit(1)
else:
    print("torch 已安装，但未注册 npu 成员。")
    sys.exit(1)
PY

echo "Ascend NPU Python 环境初始化完成。"
