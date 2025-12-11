# MindSpeed + Megatron-LM + Apex 安装教程（昇腾 Ascend NPU 环境）

本教程适用于在 **昇腾（Ascend）NPU 设备** 上搭建基于 `MindSpeed`、`Megatron-LM` 和 `Apex` 的大模型训练环境。  
所有操作均在 **Linux aarch64 架构** 下完成，推荐使用 **Python 3.10**。

---

## 🧰 系统依赖安装

```bash
# 安装基础工具
yum install -y tmux patch libjpeg-turbo-devel dos2unix openblas git

# 安装指定版本的编译工具（如系统未预装）
yum install -y gcc==7.3.0 cmake==3.12.0
```



---

## 🐍 安装 Miniconda（Python 环境管理）

```bash
mkdir ~/miniconda
cd ~/miniconda/
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh

# 初始化 conda 并激活
source ~/.bashrc
```

### 创建并激活 Conda 环境

```bash
conda create -n mindspeed python=3.10
conda activate mindspeed
```

---

## 🔥 安装 PyTorch + torch_npu（昇腾适配版）

```bash
# 安装 CPU 版 PyTorch（仅用于依赖解析，实际运行由 NPU 驱动）
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 安装昇腾 NPU 适配的 torch_npu 插件
wget https://gitee.com/ascend/pytorch/releases/download/v7.0.0-pytorch2.5.1/torch_npu-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```


---

## ⚙️ 设置昇腾环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
```

建议将上述两行加入 `~/.bashrc` 以持久生效：

```bash
echo 'source /usr/local/Ascend/ascend-toolkit/set_env.sh' >> ~/.bashrc
echo 'source /usr/local/Ascend/nnal/atb/set_env.sh' >> ~/.bashrc
```

---

## 📦 安装 MindSpeed Core

```bash
cd /data3  # 或你希望的工作目录
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 2.0.0_core_r0.8.0

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 MindSpeed
pip install -e . --no-build-isolation
```

---

# 克隆 MindSpeed-LLM（昇腾适配套件）
```bash
cd ..
git clone https://gitee.com/ascend/MindSpeed-LLM.git
```

## 🧠 集成 Megatron-LM
```bash
# 克隆官方 Megatron-LM（需能访问 GitHub）
cd ..
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cp -r megatron ../MindSpeed-LLM/  


# 切换到匹配版本
cd ../MindSpeed-LLM 
git checkout 2.0.0

# 安装其余依赖
pip install -r requirements.txt
```

> ✨ 后续训练脚本请在 `MindSpeed-LLM` 目录中运行。

---

## 🛠️ 编译安装 Apex（昇腾适配版）

```bash
# 安装兼容版本的 setuptools
pip install setuptools==65.7.0
cd ..
# 克隆并编译 Ascend Apex
git clone https://gitee.com/ascend/apex.git
cd apex
bash scripts/build.sh --python=3.10

# 强制重装生成的 wheel 包
pip install --force-reinstall apex/dist/apex-0.1+ascend-cp310-cp310-linux_aarch64.whl
```
---

## 📊 安装 TensorBoard（用于训练监控）

```bash
pip install tensorboard
```

---

## 🧪 验证安装

```bash
conda activate mindspeed
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
python -c "import apex; print('Apex loaded')"
python -c "import mindspeed; print('MindSpeed loaded')"
```

预期输出：
```
True
Apex loaded
MindSpeed loaded
```

---

## 📌 使用建议

- 使用 `tmux` 管理长时间训练任务：
  ```bash
  yum install tmux
  tmux -u new -s train
  conda activate mindspeed
  # 运行训练脚本...
  ```
- 所有训练代码应在 `MindSpeed-LLM` 目录下进行
