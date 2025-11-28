# CIFAR10简单分类网络训练/大模型推理

## 实验目标与项目介绍

本实验旨在帮助学生熟悉pytorch相关内容，熟悉模型。通过本实验，学生将学习如何处理数据、搭建简单网络，以及对推理有一定的理解。
本项目的QWEN部分实验代码在 [qwen-ascend-llm](https://github.com/Tlntin/qwen-ascend-llm.git) 上改造而成，针对本次实践进行优化。
## 第一步: Conda安装

安装 Conda
```bash
cd /data
bash Miniconda3-latest-Linux-aarch64.sh
```
重启终端

```bash
exec bash
```
# 第二步: 环境配置

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n llm_env python=3.10 -y
```

启动环境

```bash
conda activate llm_env
cd /home/test_student/
git clone https://github.com/MILVLG/engineering-practices-of-llms.git
cd engineering-practices-of-llms/
pip install -r requirements.txt 
```


# 第三步: 启动jupyter notebook 以及 npu监控

port请随意选择，例如8888。不冲突即可。
```bash
jupyter notebook --allow-root --port=8888
```

如果你使用的是简单的ssh，端口不会直接转发到你的本机。请另开终端，将端口转发
```bash
ssh -L jupyter端口号:localhost:jupyter端口号 用户名@公网ip -p 用户端口
```


另开一个终端，用于观察npu情况
```bash
watch -n 0.1 -d npu-smi info
```

# 第四步: 移动所需数据集

使用以下命令将准备好的数据移动到工作路径下

```bash
 cp -r /data/engineering-practices-of-llms/experiment0/data /home/test_student/engineering-practices-of-llms/ep0
```
