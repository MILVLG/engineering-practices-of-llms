# 实验二　大模型的微调与评估（NVIDIA GPU 版）

在四张 NVIDIA GPU 上，从 MNN-Pocket 的预训练权重开始，用 ms-swift 完成全参数指令微调。每隔半个 epoch 保存一次权重，然后用 EvalScope 比较六个检查点的数学推理、指令遵循和代码生成表现。

- **基座模型**：[taobao-mnn/MNN-Pocket-0.3B-Base](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-Base)
- **训练数据**：[Dolci 单轮指令微调子集](https://huggingface.co/datasets/Fangzhewei/dolci-instruct-sft-pocket-4096-single-turn-no-reasoning)。使用仓库中已过滤的 13 个 Parquet 分片，共 1,231,406 条样本，最长 4,096 token。

## 第一步　获取课程仓库并选择四张 GPU

以下命令在同一个 Bash 终端中执行。先用 `nvidia-smi` 查看机器上分配给自己的 GPU。将示例编号替换为自己获配的四张卡；调度系统已经设置 `CUDA_VISIBLE_DEVICES` 时直接沿用。

```bash
git clone https://github.com/MILVLG/engineering-practices-of-llms.git
cd engineering-practices-of-llms
export EP2_LESSON="$PWD/ep2/nvidia"
export EP2_WORKDIR="$PWD/ep2-work"  # 可改为有足够空间的数据盘目录
mkdir -p "$EP2_WORKDIR"/{runs,evaluations,.cache/tmp}
cd "$EP2_WORKDIR"
nvidia-smi
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
```

## 第二步　安装环境

以下命令以 Ubuntu 22.04、Python 3.10 和支持 CUDA 12.4 的 NVIDIA 驱动为例。系统依赖包括 `git`、`curl`、`python3.10-venv`、`python3.10-dev`、`build-essential`、`bubblewrap` 和 `libseccomp2`。HumanEval 用 Bubblewrap 在隔离环境中执行模型生成的代码。

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install ms-swift==4.6.1 evalscope==1.14.1 transformers==4.57.6 huggingface-hub==0.36.2 datasets==4.8.4 swanlab nltk pyarrow
python -c 'import torch; print(torch.__version__, torch.cuda.device_count()); assert torch.cuda.device_count() == 4'

# GSM8K 与 IFEval 单独使用 vLLM 环境
python3.10 -m venv .cache/venvs/pocket-vllm
.cache/venvs/pocket-vllm/bin/pip install vllm==0.8.5.post1 transformers==4.52.4
export NLTK_DATA="$PWD/.cache/nltk-data"
python -m nltk.downloader -d "$NLTK_DATA" punkt punkt_tab
python "$EP2_LESSON/evalscope_humaneval.py" --self-test
```

## 第三步　配置 Hugging Face 镜像并下载模型与数据

在能直连 Hugging Face 的机器上可使用官方端点 `https://huggingface.co`。使用国内镜像时，先设置 `HF_ENDPOINT`，再执行下载。镜像如尚未同步对应版本，可切回官方端点重试。[镜像使用说明](https://hf-mirror.com/)

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME="$EP2_WORKDIR/.cache/huggingface"
export HF_HUB_DISABLE_XET=1
hf download taobao-mnn/MNN-Pocket-0.3B-Base \
  --revision 95184b69adca5614db311c47b15fb4c447f92f84 --local-dir models/MNN-Pocket-0.3B-Base
hf download Fangzhewei/dolci-instruct-sft-pocket-4096-single-turn-no-reasoning \
  --repo-type dataset --local-dir datasets/dolci
(cd datasets/dolci && sha256sum -c SHA256SUMS)
```

## 第四步　试跑并启动全量微调

先执行两步试跑，验证四卡通信、分词和权重保存。正式训练启用 SwanLab，请输入自己账号的 API Key。运行期间可在另一终端执行 `watch -n 1 nvidia-smi` 查看显存和利用率。

```bash
stamp=$(date +%Y%m%d-%H%M%S)
run="runs/full-$stamp"
smoke="runs/smoke-$stamp"
mkdir -p "$smoke" "$run"
bash "$EP2_LESSON/train.sh" smoke "$smoke"
# 确认试跑成功后执行
read -rsp 'SwanLab API key: ' SWANLAB_API_KEY; echo
export SWANLAB_API_KEY
bash "$EP2_LESSON/train.sh" train "$run"
```

训练采用 BF16 混合精度和 `qwen3_nothinking` 模板。每卡每次读取 2 条样本，累积 16 次梯度，全局 batch size 为 4 × 2 × 16 = 128。训练 3 个 epoch，峰值学习率为 1e-5，前 5% 的步数预热，随后余弦衰减。权重保存在 `$run/checkpoints/checkpoint-*`，日志保存在同目录的 `logging.jsonl`。

## 第五步　评测六个检查点

训练完成后，选一张空闲卡串行评测。下面沿用所选四张卡中的第一张；端口 18610 如被占用，可换为空闲端口。首次运行需联网获取三个基准的数据。

```bash
gpu="${CUDA_VISIBLE_DEVICES%%,*}"
out=$(mktemp -d "$EP2_WORKDIR/evaluations/run-XXXXXX")
(
  set -e
  for ckpt in "$run"/checkpoints/checkpoint-*; do
    step="${ckpt##*-}"
    bash "$EP2_LESSON/evaluate-local-api.sh" "$out" "$run" "$gpu" "$step" 18610
    bash "$EP2_LESSON/evaluate-humaneval.sh" "$out" "$run" "$gpu" "$step"
  done
)
find "$out" -path '*/reports/*' -name '*.json' | sort -V
```

GSM8K 报告 1,319 题的答案准确率；IFEval 报告 541 条提示的 Prompt Strict，即一条提示中的所有约束都满足才算通过；HumanEval 使用 Chat 提示、贪心解码和隔离测试，报告 164 道编程题的 pass@1。

## 实践作业

1. **实验配置**。记录 GPU 型号和数量、软件版本、模型和数据集版本、训练时长及有效 batch size 的计算过程。
2. **训练曲线**。提交 SwanLab 链接或导出的曲线图，至少包含 loss、learning rate 和 grad norm。横轴标明训练步数，说明学习率预热结束前后 loss 的变化，并结合上下文解释一次明显波动。
3. **评测分数**。提交六行结果表，每行列出 epoch、checkpoint 步数、GSM8K accuracy、IFEval Prompt Strict 和 HumanEval pass@1，并附 EvalScope 报告路径。
4. **结果分析**。结合分数和一道真实评测样例说明你会选择哪个检查点，以及选择它的依据。
