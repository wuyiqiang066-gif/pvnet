#!/bin/bash
# PVNet 环境激活脚本
# 用法: source env_pvnet.sh

# 清除被 ROS2 等污染的 PYTHONPATH
export PYTHONPATH=""

CONDA_ENV_DIR="/home/cetc2028/miniconda3/envs/pvnet"
export PATH="${CONDA_ENV_DIR}/bin:$PATH"
export CONDA_PREFIX="${CONDA_ENV_DIR}"
export CUDA_HOME="/usr/local/cuda-13.0"

# 扩展库路径（torch/nvidia pip 库 + extend_utils 依赖的 ceres/glog）
NVIDIA_LIB_DIR="${CONDA_ENV_DIR}/lib/python3.12/site-packages/nvidia"
PVNET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${NVIDIA_LIB_DIR}/cublas/lib:${NVIDIA_LIB_DIR}/cuda_runtime/lib:${NVIDIA_LIB_DIR}/cusolver/lib:${NVIDIA_LIB_DIR}/cudnn/lib:${NVIDIA_LIB_DIR}/cufft/lib:${NVIDIA_LIB_DIR}/curand/lib:${NVIDIA_LIB_DIR}/cusparse/lib:${NVIDIA_LIB_DIR}/nccl/lib:${NVIDIA_LIB_DIR}/nvjitlink/lib:${CONDA_ENV_DIR}/lib/python3.12/site-packages/torch/lib:/usr/local/cuda-13.0/lib64:${PVNET_ROOT}/lib/utils/extend_utils/lib:${LD_LIBRARY_PATH}"

echo "=== PVNet Environment ==="
echo "Python: $(which python)"
