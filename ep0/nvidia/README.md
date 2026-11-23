# CIFAR10 简单分类网络训练 / 大模型推理（NVIDIA GPU 版）

## 实验目标与项目介绍

本实验旨在帮助学生熟悉 PyTorch 基础流程，学习数据处理、简单卷积网络搭建（CIFAR-10 训练与推理）以及大模型推理（Qwen KV-Cache 与采样策略）。
本目录为 **NVIDIA GPU（CUDA）** 适配版本，包含：
- `CIFAR10/exp1.ipynb`：CIFAR-10 简单分类网络训练与推理
- `Qwen_Inference/exp2_cuda.ipynb`：Qwen2.5 大模型 CUDA 推理（调用同目录 `cuda_inference.py`）

---

## 第一步：Conda 安装

检查 NVIDIA 驱动与 CUDA 版本：

```bash
nvidia-smi
```

若尚未安装 Conda，下载并安装 Linux x86_64 版 Miniconda：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
```

重启终端使 `conda` 生效：

```bash
exec bash
```

---

## 第二步：环境配置

创建 `ep0` 环境并拉取代码：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n ep0 python=3.10 -y
conda activate ep0

git clone https://github.com/MILVLG/engineering-practices-of-llms.git
cd engineering-practices-of-llms/
```

安装 CUDA 版 PyTorch 与实验依赖（以 CUDA 12.4 为例，可根据 `nvidia-smi` 显示的驱动版本调整）：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.45.2 huggingface-hub==0.36.2 accelerate sentencepiece \
    einops protobuf tiktoken numpy matplotlib tqdm pillow jupyter ipykernel
python -m ipykernel install --user --name ep0 --display-name "Python (ep0)"
```

验证 GPU 可用性：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 第三步：启动 Jupyter Notebook 与 GPU 监控

启动 Jupyter Notebook（端口可自选，例如 `8888`，不冲突即可）：

```bash
conda activate ep0
cd engineering-practices-of-llms/ep0/nvidia
jupyter notebook --allow-root --no-browser --port=8888
```

若通过 SSH 连接远程服务器，在**本地终端**转发端口后用浏览器打开终端输出的链接：

```bash
ssh -L 8888:localhost:8888 用户名@服务器IP -p SSH端口
```

另开一个终端，实时监控 GPU 显存与利用率：

```bash
watch -n 0.5 -d nvidia-smi
```

---

## 第四步：准备数据集与模型

数据与模型统一放置在 `ep0/nvidia/data/` 目录下：

- **CIFAR-10 数据集**：如服务器已有预置数据可直接复制到 `ep0/nvidia/data/`；否则在 `CIFAR10/exp1.ipynb` 中将 `download=False` 改为 `download=True` 自动下载。
- **Qwen2.5-3B-Instruct 权重**（使用国内镜像 `hf-mirror.com` 加速）：

```bash
mkdir -p ep0/nvidia/data
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    local_dir='./ep0/nvidia/data/Qwen2.5-3B-Instruct',
)
"
```

---

## 第五步：在 Notebook 中运行实验

在 Jupyter 页面右上角选择 `Python (ep0)` 内核，按顺序运行：

1. **实验一（CIFAR-10 分类网络）**：打开 `CIFAR10/exp1.ipynb`，补全 `TODO 1`–`TODO 9` 后依次运行数据加载、训练保存与加载权重推理。
2. **实验二（Qwen 大模型推理）**：打开 `Qwen_Inference/exp2_cuda.ipynb`，依次运行基础问答、Greedy / Top-K / Top-P 采样对比与不同长度测试，并结合 `nvidia-smi` 观察显存占用与释放。
