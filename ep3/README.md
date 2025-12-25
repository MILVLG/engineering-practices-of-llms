# 实验三：小型LLM后训练与评估 (RLVR/GRPO)

## 实验目标与项目介绍

本实验旨在帮助学生掌握大语言模型（LLM）后训练（Post-training）中的强化学习（RL）技术和流程。通过本实验，学生将学习如何基于 **GRPO (Group Relative Policy Optimization)** 算法，对基座模型进行对齐训练，使其具备类似 DeepSeek-R1-Zero 的推理能力（Aha-Moment）。

实验内容涵盖了环境搭建、权重转换、数据预处理、Reward（奖励）规则设计、分布式强化学习训练执行以及模型能力评估。

本项目的实验代码基于 [MindSpeed-RL](https://gitcode.com/Ascend/MindSpeed-RL) 构建，复现了 DeepSeek-R1-Zero 在 Qwen2.5-7B 上的数学推理增强效果。

## 实验依赖与环境配置

本项目的基本依赖情况如下表所示（基于商分 2.1.0 版本配套）：

| 加速卡型号 | 驱动和CANN版本 | Python版本 | 主要Python包依赖 | MindSpeed-RL版本 |
|------------|----------------|------------|------------------|------------------|
| 昇腾910B   | Ascend HDK 25.3.0，**CANN 8.2.RC1** | Python 3.10  | torch 2.5.1，torch-npu 2.5.1，ray 2.42.1 | 分支 2.1.0 |

### 1. 容器环境

```
docker images #找到id
BaseImage='' # 'a5fb032546cd' 

# 容器名字带上你的名字，避免冲突
My_Container_Name='' # zhangsan_GRPO 

# 挂载你的个人工作目录
WorkSpace='' # '/data1:/data1' '/data2:/data2'

# 2. 启动容器 (直接用 BaseImage)
docker run -itd \
  --name ${My_Container_Name} \
  --net=host \
  --shm-size=500g \
  --privileged \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v ${WorkSpace} \
  $BaseImage \
  /bin/bash

【容器外运行补丁1】
# 把宿主机的 npu-smi 强行复制到容器的 /usr/local/bin 目录下
docker cp /usr/local/sbin/npu-smi $My_Container_Name:/usr/local/bin/npu-smi
# 给容器里的这个文件加上执行权限 (以防万一)
docker exec -u 0 $My_Container_Name chmod +x /usr/local/bin/npu-smi


# 3. 进入容器
docker exec -it $My_Container_Name bash

【容器内运行补丁2】
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
npu-smi info
echo 'export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. 软件安装与源码准备

进入容器后，请参考
[安装指南](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/install_guide.md#%E5%AE%89%E8%A3%85%E6%8C%87%E5%8D%97)配置 MindSpeed-RL 2.1.0 环境：
CANN已安装cann-8.2.rc1
部分安装代码如下
```
# 安装conda
yum install wget
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh
bash Miniconda3-latest-Linux-aarch64.sh
conda create -n YOUR_ENV_NAME python==3.10
conda activate YOUR_ENV_NAME

# 安装tmux
yum install tmux
tmux 
# ctrl + b + d 退出
# tmux a -t n 进入第n个
```



### vllm及相关依赖安装：
（注：环境中需要安装git，因为vllm的安装过程依赖git）
```shell
# pydantic高版本包会产生冲突，指定版本安装
pip install pydantic==2.12.0
git clone -b releases/v0.9.1 https://github.com/vllm-project/vllm.git
cd vllm
git checkout b6553be1bc75f046b00046a4ad7576364d03c835
VLLM_TARGET_DEVICE=empty pip install .
cd ..
```

### vllm_ascend安装
```shell
git clone -b v0.9.1-dev https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 8c7bc45
pip install -r requirements.txt
pip install -e .
cd ..
```

### ray安装

```shell
pip install ray==2.42.1
```
```shell
# ray 生成的日志文件夹权限修改
# 此处针对 ray==2.42.1 实现
RAY_PATH=$(python -c "import ray; print(ray.__file__)")
UTILS_PATH=$(dirname "$RAY_PATH")"/_private/utils.py"
sed -i 's/os.chmod(\(.*\), 0o0777)/os.chmod(\1, 0o0750)/g' "$UTILS_PATH"
```

### PyTorch框架安装
（注：[PyTorch框架和torch_npu插件安装教程](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html)；可从[PyTorch-Ascend官方代码仓](https://gitcode.com/Ascend/pytorch/releases)获取PyTorch各个版本对应的torch_npu的whl包）
```shell
# 安装torch和torch_npu
#pip install torch-2.5.1-cp310-cp310-*.whl
#pip install torch_npu-2.5.1.*.manylinux2014_*.whl
```
#### 下载torch软件包和安装命令
```
pip install torch==2.5.1
```
<!-- ```
yum install wget
wget https://download.pytorch.org/whl/cpu/torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch-2.5.1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
``` -->

#### 下载torch_npu软件包和安装命令
```
pip install torch_npu==2.5.1.post1
```
<!-- ```
wget https://gitcode.com/Ascend/pytorch/releases/download/v7.1.0-pytorch2.5.1/torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip3 install torch_npu-2.5.1.post1-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
``` -->
<!-- https://www.hiascend.com/document/detail/zh/canncommercial/700/envdeployment/instg/instg_0050.html -->
### apex for Ascend 构建参考 https://gitcode.com/Ascend/apex
对于本实验环境
```
| CANN版本 | PyTorch版本 | Ascend Extension for PyTorch版本 | Python版本 | Apex 版本或代码分支 |
|:--------|:--------- |:-------------------------------|:--------|:------------------|
| 8.2.RC1  | 2.5.1      | v2.5.1-7.1.0                   | Python3.10x | master |

# 安装依赖
yum install -y patch libjpeg-turbo-devel dos2unix openblas git 
yum install -y gcc==10.3.1 cmake==4.2.0

# 请确保已安装PyTorch框架且setuptools版本小于等于65.7.0，若版本不符合条件，可使用以下命令安装
pip install setuptools==65.7.0
# 获取昇腾适配的Apex-patch源码
git clone -b master https://gitcode.com/Ascend/apex.git
cd apex/
# 执行
bash scripts/build.sh --python=3.10
# pip install apex-0.1.dev*.whl
cd apex/dist/
pip3 uninstall apex
pip3 install --upgrade apex-0.1+ascend-{version}.whl  # version为Python版本和CPU架构
```




<!-- ```bash
# 1. 安装基础依赖
pip install ray==2.42.1
pip install pydantic==2.12.0
# 修正 Ray 权限
RAY_PATH=$(python -c "import ray; print(ray.__file__)")
UTILS_PATH=$(dirname "$RAY_PATH")"/_private/utils.py"
sed -i 's/os.chmod(\(.*\), 0o0777)/os.chmod(\1, 0o0750)/g' "$UTILS_PATH" -->
### 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
### OpenEuler 操作系统

执行如下命令重操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jemalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
#根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量(可通过 find /usr -name libjemalloc.so.2 确认文件是否存在)
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2
```



# 2. 准备源码 (使用 2.1.0 配套版本)
```shell
git clone https://gitcode.com/Ascend/MindSpeed-RL.git

git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 89f4632d2cbb1e583a69b2bf3a08d75222f1173d  # 参考MindSpeed-LLM依赖版本
pip install -r requirements.txt 
cp -r mindspeed ../MindSpeed-RL/
cd ..

# Megatron从github下载，请确保网络能访问
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-RL/
cd ..

git clone https://gitcode.com/Ascend/MindSpeed-LLM.git -b 2.1.0
cd MindSpeed-LLM
git checkout 887c2d8682021befd675bb03965dbdee4de24516
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

cd ./MindSpeed-RL
pip install -r requirements.txt
pip install antlr4-python3-runtime==4.9.3 --no-deps 
```
<!-- git clone -b 2.1.0 https://gitcode.com/Ascend/MindSpeed-RL.git

# 准备 MindSpeed-LLM (依赖)
git clone -b 2.1.0 https://gitcode.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
# 安装依赖
pip install -r requirements.txt
cp -r mindspeed_llm ../MindSpeed-RL/
cd ..

# 准备 Megatron-LM (Core 0.8.0)
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.8.0
cp -r megatron ../MindSpeed-RL/
cd ..

# 3. 安装 MindSpeed-RL
cd MindSpeed-RL

git checkout tags/2.1.0 -b v2.1.0
pip install -r requirements.txt
``` -->

## 实验设计与指导

### 实验设定与流程

我们对小型LLM的强化学习后训练任务进行了如下设定：
- **模型设定**：使用 `Qwen2.5-7B` 作为 Base 模型。该模型指令遵从度高，适合作为 R1-Zero 流程的起点。
- **算法设定**：采用 **GRPO (Group Relative Policy Optimization)**。与 PPO 不同，GRPO 不需要额外的 Value Network (Critic)，通过分组采样和组内优势估计来优化策略，节省显存并提升训练效率。
- **数据集**：使用 `DeepScaler-Preview-Dataset` (40K条数学推理数据)。
- **研究性任务**：观察训练过程中的 "Aha-Moment"（顿悟时刻），即模型开始自发生成 `<think>` 标签并进行自我修正的过程。

<!-- 请参考下图工作流合理分配小组工作，安排实验进度：

![实验工作流](../../sources/images/r1_zero/r1_zero_roadmap.png) -->

### 模型下载
1.  **使用hf-mirror**：进入[hf-mirror]{https://hf-mirror.com/},使用 huggingface-cli
```
1. 安装依赖
pip install -U huggingface_hub

2. 设置环境变量
Linux
export HF_ENDPOINT=https://hf-mirror.com
```
2.  **下载模型**：[Qwen2.5-7B]{https://hf-mirror.com/Qwen/Qwen2.5-7B}。
```
huggingface-cli download --resume-download Qwen/Qwen2.5-7B --local-dir  Qwen2.5-7B
```

### 数据集准备与预处理

1.  **下载数据**：下载 DeepScaler 数据集与gsm8k 数据集。分别放置于 `dataset/deepscaler.json` 和 `dataset/gsm8k/test.jsonl`
2.  **配置模板**：修改 `configs/datasets/deepscaler.yaml` 中确认数据映射。
```shell
input: ./deepscaler.json
tokenizer_name_or_path: ./Qwen2.5-7B/
output_prefix: ./dataset/deepscaler/data
```
3.  **执行预处理**：将数据转换为训练所需的格式，并添加 R1 风格的 Prompt 模板。

```bash
# 在 MindSpeed-RL 目录下执行
cd MindSpeed-RL
mkdir -p ./dataset/deepscaler
bash examples/data/preprocess_data.sh deepscaler
```
*注：Prompt 模板会自动包裹 `<think>` 和 `<answer>` 标签引导模型输出。*

### 模型权重转换

MindSpeed-RL 基于 Megatron 架构，需要将 HuggingFace 格式的权重转换为 Megatron 格式。
MindSpeed-LLM 下提供了转换代码。修改路径和配置信息。
```bash
cd ../Megatron-LM
git checkout core_v0.12.0
# rm -rf MindSpeed-LLM/megatron
cd ../MindSpeed
git checkout master
# rm -rf MindSpeed-LLM/mindspeed

cd ../MindSpeed-LLM
git stash
git checkout master
git stash pop
cp -r ../MindSpeed/mindspeed ./
cp -r ../Megatron-LM/megatron ./
pip install peft
bash examples/mcore/qwen25/ckpt_convert_qwen25_hf2mcore.sh
```

### 模型后训练 (GRPO)

本实验使用基于规则的奖励模型（Rule-based Reward），不训练独立的 Reward Model。
训练步数step设置为500
1.  **配置训练参数**：
    修改 `configs/grpo_qwen25_7b_A3.yaml`。关键参数说明：
    - `megatron_training`:
        -  `tokenizer_name_or_path`: ./Qwen2.5-7B/
        -  `data_path`: ./MindSpeed-RL/dataset/data
    - `actor_config`:
        -   `load`: ./Qwen2.5-7B_mcore/  
        -   `save`: ./Qwen2.5-7B_mcore_math40k/
        -   `tensor_model_parallel_size`: 4
        -   `pipeline_model_parallel_size`: 1
    - `rl_config`:
        - `num_npus`: 8 (使用单机8卡)
    - `generate_config`:
        -   `infer_tensor_parallel_size`: 4       # 8卡推理：TP=4
    - `kl_coeff`: KL 散度系数，控制模型不偏离基座太远。
    - `n_samples_per_prompt`: 组采样个数 (G)，本实验设为8。
 
2.  **配置训练脚本**：
    修改`examples/grpo/grpo_trainer_qwen25_7b.sh`
 
    1.核心网络配置 
    脚本中有两处需要修改网络相关的变量，请确保在所有节点上根据实际情况填写。

    1.1 获取网卡名称和 IP
    在终端运行以下命令查看你的网卡名称（Interface Name）和 IP 地址：
    ```bash
    yum install -y net-tools iproute
    ```
    运行
    ```
    ifconfig
    # 或
    python -c "import socket; import fcntl; import struct; import array; 
    def get_interfaces():
        max_ifaces = 32; bytes = max_ifaces * 32
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        names = array.array('B', b'\0' * bytes)
        outbytes = fcntl.ioctl(s.fileno(), 0x8912, struct.pack('iL', bytes, names.buffer_info()[0]))[0]
        namestr = names.tobytes(); lst = []
        for i in range(0, outbytes, 40):
            name = namestr[i:i+16].split(b'\0', 1)[0].decode()
            try:
                ip = socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', name[:15].encode('utf-8')))[20:24])
                lst.append((name, ip))
            except: pass
        return lst
    print('Found interfaces:'); 
    for name, ip in get_interfaces(): print(f'Name: {name} | IP: {ip}')"
    ```
    找到对应内网通信 IP 的网卡名称（例如：eth0）。

    1.2 修改脚本变量
    在脚本中找到以下几行并进行修改：
    ```bash
    # [修改点 1] 设置主节点 IP 地址
    # 单机训练，这里填本机 IP
    MASTER_ADDR="x.x.x.x"  # <--- 修改这里

    # [修改点 2] 设置当前节点的通信网卡名称
    # 用于 Ray 获取本机 IP 进行节点注册
    SOCKET_IFNAME="eth0"         # <--- 修改这里 (填你通过 ifconfig 查到的网卡名)

    # 注意：脚本开头有一行 SOCKET_IFNAME="Your SOCKET IFNAME"，但脚本中间又重新定义了一次 SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"。建议直接修改中间那一行，或者确保两处一致。
    
    # [修改点 3] 设置节点卡数
    NPUS_PER_NODE=8

    #  [修改点 4]请确认该文件存在(可通过 find /usr -name libjemalloc.so.2 确认)
    export LD_PRELOAD=/usr/lib64/libjemalloc.so.2
    export LD_PRELOAD=YOUR_PATH
    ```

2.  **启动 Ray 集群**：
    GRPO 训练依赖 Ray 进行分布式调度。
    ```bash
    # 在主节点启动 Ray head
    export MASTER_ADDR=localhost
    ray start --head --port 6344 --dashboard-host=$MASTER_ADDR --dashboard-port=8260
    ```

3.  **启动训练**：
    ```bash
    # 设置环境变量
    export HCCL_CONNECT_TIMEOUT=1800
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    
    # 启动训练脚本
    chmod 7 examples/grpo/grpo_trainer_qwen25_7b.sh
    bash examples/grpo/grpo_trainer_qwen25_7b.sh
    ```


### 模型部署
可以进行简单的对话
```bash
cd MindSpeed-LLM
bash examples/mcore/qwen25/generate_qwen25_7b_ptd.sh
```

### 模型评估

训练完成后，使用数学评测集（gsmk8k）子集进行评估。
复制同目录gsm8k_eval.py覆盖原文件
```
cp ./ep3/gsm8k_eval.py ./mindspeed_llm/tasks/evaluation/eval_impl/gsm8k_eval.py
```
复制同目录test.jsonl到数据目录下
```
cp ./ep3/test.jsonl ./dataset/gsm8k/test.jsonl
```
之后修改脚本
```bash
CHECKPOINT="./Qwen2.5-7B_mcore/"
TOKENIZER_PATH="./Qwen2.5-7B/"
DATA_PATH="./dataset/gsm8k/" # 对应./dataset/gsm8k/test.jsonl
TASK="gsm8k"
```
然后
```bash
bash examples/mcore/qwen25/evaluate_qwen25_7b_ptd.sh
```

### 消融实验 (选做)

针对 GRPO 训练过程，选择以下任一维度进行消融实验：
- **Reward 设计**：修改 `mindspeed_rl/reward/reward_rules.py`，调整格式奖励（Format Reward）和答案准确性奖励（Accuracy Reward）的权重，观察对收敛速度的影响。
- **采样数量 (G)**：调整 GRPO 的 `num_generations` (例如从 4 改为 8)，分析显存占用与训练效果的权衡。
- **迭代步数**：对比 200 iter 和 400 iter 的模型在 gsm8k 上的性能差异。

## 实践作业提交内容
1.  **环境截图**：Docker 容器启动成功及 `pip list` 包含 mindspeed-rl 的截图。
2.  **训练日志**：
    - 提供训练过程中的 Loss 曲线图（TensorBoard 截图或 Log 数据）。
    - 提供 Reward 变化曲线图（证明模型学到了规则）。
3.  **Aha-Moment 样例**：
    - 截取一个训练后的模型输出 Case，展示其 `<think>` 标签内的思考过程（特别是自我修正或长链推理的部分）。
4.  **实验报告**：
    - 内容包括但不限于实验经过记录、训练与评估结果分析、消融实验结果与分析等
