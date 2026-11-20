# 训练可视化脚本使用说明

## 环境依赖
- Python 3.10+
- `matplotlib`（已通过清华源安装：`pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib`）

> 所有命令默认在 `m2` 虚拟环境中执行。

## 1. plot_training_curves.py
位置：`scripts/plot_training_curves.py`

功能：
- 解析一个或多个 MindSpeed 训练日志，自动抓取 `seq_length`、`global batch size`、train/val loss。
- 横轴根据 token 数（`global_batch_size * batch_seq_len * step` 或 `consumed samples * seq_len`）换算。
- 绘制平滑后的 train loss（实线）与 raw train loss（alpha=0.5），以及虚线标记的 val loss，便于同图对比多次实验。

常用参数：
- `--output`: 输出图片路径，默认 `outputs/training_curves.png`。
- `--smooth-window`: train loss 平滑窗口，默认 50。
- `--title`: 图标题。
- `--seq-length`: 若日志缺少 seq length，可手动指定。

示例：
```bash
python scripts/plot_training_curves.py \
  logs/will_500m_ptd_20251113_000009.log \
  logs/will_500m_ptd_20251113_200319.log \
  --output outputs/demo_training_curves.png \
  --smooth-window 100
```

## 2. plot_exp_loss_after_tokens.py
位置：`scripts/plot_exp_loss_after_tokens.py`

功能：
- 重用统一的日志解析器，只保留达到某个 token 阈值之后的数据点。
- 将 loss 转换为 `exp(loss)`（≈ ppl）并绘图，帮助聚焦高 token 区间的收敛行为。
- 支持同时比较多个实验，train 曲线展示平滑与原始两条，val 曲线以虚线+标记显示。

常用参数：
- `--token-threshold`: 起始 token 数，默认 `5e9`（5B）。
- 其余参数与 `plot_training_curves.py` 一致：`--output`、`--smooth-window`、`--title`、`--seq-length`、`--dpi`。

示例：
```bash
python scripts/plot_exp_loss_after_tokens.py \
  logs/will_500m_ptd_20251113_000009.log \
  logs/will_500m_ptd_20251113_200319.log \
  --token-threshold 5e9 \
  --smooth-window 100 \
  --output outputs/exp_loss_after_5B.png
```

## 图像输出位置
- 所有示例命令默认把图像写入 `outputs/` 目录，可根据需要传入 `--output` 定制。
- 若目录不存在，脚本会自动创建。

## 调试技巧
- 解析日志失败时，可用 `--seq-length` 明确 batch 序列长度。
- 如需更平滑的曲线，可调大 `--smooth-window`；若想观察原始波动，可调小甚至设为 1。
- 支持传入任意数量的日志文件，可快速评估不同训练脚本或超参组合的效果。
