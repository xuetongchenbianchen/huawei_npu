#!/bin/bash
# 一键检查Ascend软件安装情况脚本
# 作者: AI硬件开发者助手

echo -e "\033[1;34m=========================================="
echo -e "      Ascend 软件安装环境检查报告"
echo -e "==========================================\033[0m"

# 1. 检查操作系统
echo -e "\n\033[1;33m1. 操作系统信息:\033[0m"
cat /etc/os-release | grep -E "NAME|VERSION"

# 2. 检查驱动 (Driver)
echo -e "\n\033[1;33m2. 驱动 (Driver) 状态:\033[0m"
if command -v npu-smi &> /dev/null; then
    echo -e "\033[1;32m[OK] npu-smi 工具已安装。\033[0m"
    echo -e "当前设备状态:"
    npu-smi info
else
    echo -e "\033[1;31m[ERROR] npu-smi 未找到。驱动可能未安装或未配置环境变量。\033[0m"
fi

# 3. 检查固件 (Firmware)
echo -e "\n\033[1;33m3. 固件 (Firmware) 状态:\033[0m"
if [ -f /usr/local/Ascend/driver/version.info ]; then
    DRV_VER=$(cat /usr/local/Ascend/driver/version.info | grep "Driver Version" | awk '{print $3}')
    echo -e "驱动版本: $DRV_VER"
    
    # 尝试读取固件版本 (不同版本路径可能不同)
    if [ -f /usr/local/Ascend/driver/firmware/version.info ]; then
        FW_VER=$(cat /usr/local/Ascend/driver/firmware/version.info | grep "Firmware Version" | awk '{print $3}')
        echo -e "固件版本: $FW_VER"
    else
        echo -e "\033[1;33m[WARN] 无法直接读取固件版本文件，建议使用 npu-smi info 查看。\033[0m"
    fi
else
    echo -e "\033[1;31m[ERROR] 驱动版本文件不存在。\033[0m"
fi

# 4. 检查 CANN 包 (Toolkit/NNRT)
echo -e "\n\033[1;33m4. CANN 软件包 (Ascend Toolkit/NNRT) 检查:\033[0m"

# 检查环境变量
if [ -z "$ASCEND_HOME" ]; then
    echo -e "\033[1;33m[WARN] ASCEND_HOME 环境变量未设置。\033[0m"
    # 尝试自动探测
    if [ -d "/usr/local/Ascend/ascend-toolkit" ]; then
        ASCEND_HOME="/usr/local/Ascend"
        echo -e "自动探测到 CANN 安装路径: $ASCEND_HOME"
    elif [ -d "/usr/local/Ascend/nnrt" ]; then
        ASCEND_HOME="/usr/local/Ascend"
        echo -e "自动探测到 NNRT 安装路径: $ASCEND_HOME"
    else
        echo -e "\033[1;31m[ERROR] 未找到 CANN 安装目录。\033[0m"
    fi
else
    echo -e "\033[1;32m[OK] ASCEND_HOME: $ASCEND_HOME\033[0m"
fi

# 4.1 检查 ccec 编译器版本
echo -e "\n\033[1;33m4.1 ccec 编译器版本:\033[0m"
if command -v ccec &> /dev/null; then
    ccec --version
elif command -v ccec_compiler &> /dev/null; then
    ccec_compiler --version
else
    echo -e "\033[1;33m[WARN] 未找到 ccec/ccec_compiler，可检查 PATH 是否包含 CANN 工具路径。\033[0m"
fi

# 检查版本文件
if [ -d "$ASCEND_HOME" ]; then
    # 查找 version.info
    VER_FILE=$(find $ASCEND_HOME -name "version.info" | grep -E "toolkit|nnrt" | head -n 1)
    if [ -f "$VER_FILE" ]; then
        echo -e "CANN 版本信息:"
        cat $VER_FILE | grep -E "Version|Release"
    else
        echo -e "\033[1;33m[WARN] 未找到具体的 CANN 版本信息文件。\033[0m"
    fi
fi

# 5. 检查 Python 与 PyTorch/Ascend
echo -e "\n\033[1;33m5. Python 与 框架适配检查:\033[0m"
if command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version | awk '{print $2}')
    echo -e "Python 版本: $PY_VER"
    
    # 检查 torch_npu
    echo -e "\nPyTorch Ascend 插件检查:"
    python3 - << 'PYSCRIPT'
import importlib
import torch
try:
    import torch_npu
    print("\033[1;32m[OK] torch_npu 已安装。\033[0m")
    print("torch_npu 版本:", torch_npu.__version__)
    print("torch 版本:", torch.__version__)
    # 简单测试
    device = torch.device('npu:0')
    x = torch.randn(1, 3, 224, 224).to(device)
    print("\033[1;32m[OK] NPU 设备可用，已成功创建张量。\033[0m")
except ImportError:
    print("\033[1;33m[WARN] torch_npu 未安装。\033[0m")
except Exception as e:
    print("\033[1;31m[ERROR] NPU 初始化失败:", e, "\033[0m")
PYSCRIPT
else
    echo -e "\033[1;31m[ERROR] python3 未找到。\033[0m"
fi

# 6. 检查 LD_LIBRARY_PATH
echo -e "\n\033[1;33m6. 库路径 (LD_LIBRARY_PATH) 检查:\033[0m"
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -i ascend

echo -e "\n\033[1;34m=========================================="
echo -e "            检查结束"
echo -e "==========================================\033[0m"
