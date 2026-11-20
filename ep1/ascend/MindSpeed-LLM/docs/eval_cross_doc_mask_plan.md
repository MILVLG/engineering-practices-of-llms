# Cross-Document Loss Mask Plan

## 背景
当前训练流程默认采用 packing：同一个序列中可能包含多条文档，以 `EOD/EOS token` 做分隔。例如：
```
... token_i, EOD, token_j_next_doc ...
```
在默认配置（未启用 `--eod-mask-loss`）下：
- `token_i -> EOD` 这一步会计算 loss；
- `EOD -> token_j_next_doc` 也会计算 loss。

即使启用 `--eod-mask-loss`，也只是屏蔽了「预测 EOD 本身」的 loss（`label == EOD` 的位置）；EOD 之后的下一步（预测下一篇文档首 token）仍然会参与训练，这一点需要特别区分，以免混淆。

## 需求
某些场景希望“跨文档衔接处不带梯度”，也就是连 `EOD -> 下一篇文档首 token` 这一格都不参与训练，以免模型在两个文档之间学习到不必要的衔接式样。当前代码没有这一逻辑，需要显式增加。

## 方案构想
1. 在生成 `loss_mask` 时，除了 `--eod-mask-loss` 控制的 `label == EOD`，再新增一个可选开关 `--mask-cross-doc`
   - 对 `EOD` 位置下一格（即 `input == EOD` 的下一个 index）置 0；
   - 实现上可在 `get_ltor_masks_and_position_ids()` 中扫描 `data == eod_token` 的索引，再对 `mask[:, i+1]` 设为 0（注意边界）。

2. 参数层面
   - 新增 `--mask-cross-doc-loss` 或类似 flag（默认 False），并在 `arguments.py`、`pretrain_gpt.py` 等入口传递；
   - 文档中明确该 flag 与 `--eod-mask-loss` 的区别：前者屏蔽 “EOD -> 下一篇文档首 token”，后者屏蔽 “token_i -> EOD”。

3. 验证
   - 构造简单样本 `[a, b, EOD, c, d, EOD, ...]`，打印 loss_mask，确认 `--mask-cross-doc-loss` 会让 `EOD` 右侧那一格为 0；
   - 同时测试 `--eod-mask-loss` 与 `--mask-cross-doc-loss` 独立/同时开启时的行为，确保不会互相干扰。

## 注意事项
- 需要强调 `--eod-mask-loss` 与 “跨文档屏蔽” 的区别，以免再出现混淆；
- 由于这个逻辑影响梯度回传，修改后需在训练日志中确认 loss 数值、统计指标不出现异常跳变。

