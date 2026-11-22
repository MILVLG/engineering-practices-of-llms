# CIFAR10 简单分类网络训练 / 大模型推理

## 实验目标与项目介绍

本实验旨在帮助学生熟悉 PyTorch 基础流程，学习数据处理、简单卷积网络搭建（CIFAR-10 训练与推理）以及大模型推理（Qwen KV-Cache 与采样策略）。
本项目 Qwen 昇腾版推理代码在 [qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm.git) 基础上改造，同时提供了独立的 NVIDIA GPU 推理版本（不覆盖原代码）。

| 项目 | 昇腾 NPU | NVIDIA GPU |
| --- | --- | --- |
| Conda 安装包 | `Linux-aarch64` | `Linux-x86_64` |
| 核心依赖 | `requirements.txt`（含 `torch-npu`） | PyTorch CUDA + `transformers` |
| 硬件监控 | `watch -n 0.1 -d npu-smi info` | `watch -n 0.5 -d nvidia-smi` |
| 实验一 Notebook | `ep0/CIFAR10/exp1.ipynb` | `ep0/CIFAR10/exp1.ipynb` |
| 实验二 Notebook | `ep0/Qwen_Inference/exp2.ipynb` | `ep0/Qwen_Inference/exp2_cuda.ipynb` |

---

## 第一步：Conda 安装

**昇腾 NPU**
```bash
cd /data
bash Miniconda3-latest-Linux-aarch64.sh
```

**NVIDIA GPU**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
```

安装完成后重启终端：
```bash
exec bash
```

---

## 第二步：环境配置

创建环境并拉取代码：
```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n ep0 python=3.10 -y
conda activate ep0

git clone https://github.com/MILVLG/engineering-practices-of-llms.git
cd engineering-practices-of-llms/
```

**昇腾 NPU**
```bash
pip install -r requirements.txt
```

**NVIDIA GPU**（以 CUDA 12.4 为例，可通过 `nvidia-smi` 查看驱动支持版本）
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.45.2 huggingface-hub==0.36.2 accelerate sentencepiece \
    einops protobuf tiktoken numpy matplotlib tqdm pillow jupyter ipykernel
python -m ipykernel install --user --name ep0 --display-name "Python (ep0)"
```

---

## 第三步：启动 Jupyter Notebook 与硬件监控

启动 Jupyter（端口可自选，例如 `8888`）：
```bash
conda activate ep0
cd engineering-practices-of-llms/ep0
jupyter notebook --allow-root --no-browser --port=8888
```

若通过 SSH 连接远程服务器，在**本地终端**转发端口后用浏览器打开链接：
```bash
ssh -L 8888:localhost:8888 用户名@服务器IP -p SSH端口
```

另开终端监控设备状态：
- **昇腾 NPU**：`watch -n 0.1 -d npu-smi info`
- **NVIDIA GPU**：`watch -n 0.5 -d nvidia-smi`

---

## 第四步：准备数据集与模型

**昇腾 NPU（使用服务器预置数据）**
```bash
cp -r /data/engineering-practices-of-llms/experiment0/data ./ep0/
```

**NVIDIA GPU（自行下载至 `ep0/data/`）**
- CIFAR-10：在 `ep0/CIFAR10/exp1.ipynb` 中将 `download=False` 改为 `download=True`。
- Qwen2.5-3B-Instruct：
```bash
mkdir -p ep0/data
HF_ENDPOINT=https://hf-mirror.com python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen2.5-3B-Instruct', local_dir='./ep0/data/Qwen2.5-3B-Instruct')
"
```

---

## 第五步：运行 Notebook

在 Jupyter 中选择 `ep0` 内核，按顺序运行：
1. **实验一（CIFAR-10）**：打开 `ep0/CIFAR10/exp1.ipynb`，补全 `TODO 1`–`TODO 9` 后依次运行训练与推理。
2. **实验二（Qwen 推理）**：
   - **昇腾 NPU**：打开 `ep0/Qwen_Inference/exp2.ipynb`
   - **NVIDIA GPU**：打开 `ep0/Qwen_Inference/exp2_cuda.ipynb`
