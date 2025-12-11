# 实验二：大模型的微调与评估
## 实验目标与项目介绍
本实验旨在帮助学生掌握大型语言模型（LLM）的全量微调和基于低秩适应（LoRA）的微调技术。通过实际操作，学生将学习如何准备微调数据集、配置微调环境、执行全量微调任务以及使用LoRA方法进行高效微调，并对微调后的模型进行评估。实验内容包括数据处理、模型配置、训练优化等关键步骤，以提升学生对大模型微调技术的理解和实践能力。

## 实验依赖与环境配置

本项目的基本依赖情况如下表所示：

| 加速卡型号 | 驱动和CANN版本 | Python版本 | 主要Python包依赖 |
|------------|----------------|------------|------------------|
| 昇腾910B   | Ascend HDK 23.0.6，CANN 8.0.RC3 | Python 3.10  | torch 2.5.1，torch-npu 2.5.1.post1，transformers 4.43.2 等 |

请参考`docs/environment.md`文件配置本次实验所需的环境。

## 实验设计与指导

### 第一步：实验环境准备(均以Qwen1.5 7b为例)
本实验的基础环境基本与实验一相同，只需要额外下载模型和数据集即可
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
huggingface-cli download Qwen/Qwen1.5-7B --local-dir 模型存放路径
```

### 第二步：模型权重转换
由于实验使用的是Ascend 910B，而Qwen2.5-7B模型是在NVIDIA GPU上预训练的，因此需要将模型权重转换为Ascend 910B兼容的格式。（hf2mcore）
```bash
# 以Qwen1.5-7B为例
# 注意修改模型路径和保存路径
cd MindSpeed-LLM
bash examples/mcore/qwen15/ckpt_convert_qwen15_hf2mcore.sh
```
关键参数
- `--target-tensor-parallel-size`：目标张量并行大小（例如8）
- `--target-pipeline-parallel-size`：目标流水线并行大小（例如1）

成功转换后的模型文件结构如下：(tp=8,pp=1为例)
```
Qwen2.5-7B-mcore/
├── iter_0000001/
    ├── mp_rank_00/
        ├── model_optim_rng.pt
    ├── mp_rank_01/
        ├── model_optim_rng.pt
    ├── mp_rank_02/
        ├── model_optim_rng.pt
    ├── mp_rank_03/
        ├── model_optim_rng.pt
    ├── mp_rank_04/
        ├── model_optim_rng.pt
    ├── mp_rank_05/
        ├── model_optim_rng.pt
    ├── mp_rank_06/
        ├── model_optim_rng.pt
    ├── mp_rank_07/
        ├── model_optim_rng.pt
├── latest_checkpointed_iteration.txt
```
```bash
# 可以验证是否模型转换成功，进行简单的对话
bash examples/mcore/qwen25/generate_qwen15_7b_ptd.sh
```

### 第三步：数据集格式转换
本实验使用的微调数据集为alpaca-chinese-52k-v3.json，

alpaca-chinese-52k-v3.json 样例展示
```json
    {
        "en_instruction": "Give three tips for staying healthy.",
        "en_input": "",
        "en_output": "1.Eat a balanced diet and make sure to include plenty of fruits and vegetables. \n2. Exercise regularly to keep your body active and strong. \n3. Get enough sleep and maintain a consistent sleep schedule.",
        "zh_instruction": "给出保持健康的三个建议。",
        "zh_input": "",
        "zh_output": "1.均衡饮食，确保多吃水果和蔬菜。\n2.定期锻炼，保持身体活跃和强壮。\n3.获得充足的睡眠并保持一致的睡眠时间表。",
        "metadata": {
            "translated": true,
            "score": 5
        }
    },
```
数据集的预处理脚本为preprocess_data.py。
```bash
python ./preprocess_data.py \
        --input 数据集路径 \
        --tokenizer-name-or-path 模型tokenizer存放路径（例如Qwen/Qwen2.5-7B） \
        --output-prefix 转换后数据集存放路径 \
        --workers 4 \ # 工作线程数
        --log-interval 1000 \ # 日志打印间隔
        --tokenizer-type PretrainedFromHF \ # 模型tokenizer类型（一般为PretrainedFromHF）
        --handler-name AlpacaStyleInstructionHandler \ # 数据集处理程序（一般为AlpacaStyleInstructionHandler）
        --prompt-type qwen \ # 模型prompt类型
        --map-keys '{"prompt": "zh_instruction", "query": "zh_input", "response": "zh_output"}' # 数据集字段映射（根据实际数据集字段修改）
```
或直接运行
```bash
bash example/mcore/qwen15/data_convert_qwen15_ft.sh
```


### 第四步 模型微调
#### 全量微调
全量微调是指对整个模型参数进行微调，包括所有层的权重。在
一般命名为 tune_模型名_大小_full_ptd.sh
以Qwen2.5-7B为例
```bash
bash example/mcore/qwen15/tune_qwen25_7b_full_ptd.sh
```
#### 基于LoRA的微调
基于LoRA的微调是指在全量微调的基础上，通过添加低秩适配器（LoRA）来减少微调参数数量，从而提高微调效率。
一般命名为 tune_模型名_大小_lora_ptd.sh
以Qwen2.5-7B为例
```bash
bash example/mcore/qwen25/tune_qwen15_7b_lora_ptd.sh
```

注意
1. tp和pp与转换的权重保持一致
2. 可以改变的超参数
    - 学习率（lr）
    - 批量大小（micro-batch-size && global-batch-size）
    - 训练轮数（train-iters）
    - 权重衰减（weight-decay）
3. 部分重要超参数解释

- 学习率相关
  - `--lr`：优化器的初始学习率。
  - `--lr-decay-style`：学习率衰减策略。`cosine` 在 SFT 中更平滑，适合不剧烈的性能调参。
  - `--min-lr`：最小学习率，通常设置为初始 LR 的 `0.1` 倍左右。若你将 `lr` 调高，请同步按比例调高。
  - `--lr-warmup-fraction`：预热比例，常用 `0.01 ~ 0.05`。长序列训练建议至少 `0.01`，避免初期不稳。

- 批量与并行
  - `--micro-batch-size`（MBS）：每个流水线阶段的局部小批大小。直接受显存影响，增大会显著增加显存占用。
  - `--global-batch-size`（GBS）：跨数据并行的全局批大小。影响每步的样本数与吞吐。
  - 有效微批数量：`num_micro_batches = GBS / (DP_SIZE * MBS)`，必须为整数。

- 训练步数
  - `--train-iters`：优化步数（非样本轮数）。训练总 token 近似为 `train-iters * GBS * 平均序列长度`。

- 正则化与稳定性
  - `--weight-decay`：权重衰减。全参 SFT 通常较小（`0.0 ~ 0.01`）；
  - `--clip-grad`：梯度裁剪阈值（当前 `1.0`），有助于避免梯度爆炸，尤其是长序列与全参更新。
  - `--adam-beta1` / `--adam-beta2`：Adam 动量参数。常见 SFT 配置为 `beta1=0.9`，`beta2=0.95` 或 `0.999`。
  - `--initial-loss-scale`：用于 FP16 的 Loss Scale 初始值；
  - `--attention-softmax-in-fp32`：提升数值稳定性，建议保留。

- 序列与模型
  - `--seq-length` 与 `--max-position-embeddings`：训练序列长度上限与位置嵌入长度。
  - `--variable-seq-lengths`：动态序列有利于减少填充、提升吞吐；建议用于指令微调。
  - `--num-layer-list` 与 `--pipeline-model-parallel-size`：按 PP 进行分层切分，保证分割列表之和等于总层数。
  - `--tensor-model-parallel-size`：张量并行。

- 模板与分词器
  - `--prompt-type qwen`：使用 Qwen 模板构造对话与指令格式，必须与数据预处理一致，否则会出现格式错配。
  - `--tokenizer-type PretrainedFromHF` 与 `--tokenizer-name-or-path`：分词器来源路径。确保与权重对应版本一致。


### 第五步 模型部署
可以进行简单的对话
```bash
bash examples/mcore/qwen25/generate_qwen25_7b_ptd.sh
```

### 第六步 模型评估
在ceval数据集上评估模型
```bash
bash examples/mcore/qwen25/evaluate_qwen25_7b_ptd.sh
bash examples/mcore/qwen25/evaluate_qwen15_7b_lora_ptd.sh
```
- 模板与分词器
  - `--task`：改为ceval
## 实践作业提交内容

- 项目输出的微调log文件和评估log文件
- 实验报告，内容包括但不限于实验经过记录、微调与评估结果分析、消融实验结果与分析等
- 加分项：自己自由选择模型和数据集进行微调训练