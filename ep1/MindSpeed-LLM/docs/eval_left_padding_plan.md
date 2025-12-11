# Evaluation Left-Padding Plan

## 背景
当前 `sample_sequence_batch` / `pad_batch` 逻辑采用右侧 padding。批次内如果存在长短差异显著的 prompt，生成循环会从 `context_length = min(context_lengths)` 开始逐 token 预测；较长的样本在前若干步只是在“占位”，参加同样的前向但并不会写入新 token，造成算力浪费。此外，在调试生成轨迹（token id/文本）时，step 编号也会从最短 prompt 的长度开始，阅读上不太直观。

## 原理
- 相对位置编码（RoPE）只关心 token 之间的相对距离。在左 padding 的场景下，只要 **attention mask** 屏蔽掉 padding，RoPE 的旋转角不会受到常量偏移的影响。
- 因此我们可以把真实 token 移到右边，左侧补 padding，并保持 position id 继续从左到右累积（或重新编号 0..N-1 均可）。
- 关键是 attention mask 必须与新的 padding 方位一致，确保左侧 padding 不参与 self-attention。

## 方案
1. **pad_batch / get_batch 支持左 padding**
   - 新增 `left_pad` 参数（默认 False），当为 True 时，将每条样本移到张量右端，左侧补 `pad_id`；
   - attention mask 改为“左侧为 True（屏蔽），右侧真实 token 为 False”；
   - position id 维持现有计算（从左到右累加），或视情况在 `left_pad` 分支中映射为 0..有效长度-1。

2. **sample_sequence_batch 起点**
   - 当前使用 `context_length = context_lengths.min()`；左 padding 后可改为 `context_length = context_lengths.max()`（或先整体偏移再递增），从最长 prompt 起步，这样所有样本都会直接进入“生成”阶段，不再空跑。

3. **调试/验证**
   - 选取包含长短 prompt 的小批次，比较左/右 padding 版本输出是否一致；
   - 观察 `Megatron.generate micro_batch`、token id 日志，确认 step 递增符合预期；
   - 确认 RoPE、max_position_embeddings 等配置不越界。

4. **渐进落地**
   - 优先在 `scripts/run_all_evals_exp.sh` 这种实验脚本里开启 `left_pad`，验证性能收益；
   - 如运行稳定，再推广到主评测脚本。

## 后续
- 该文档记录左 padding 方案方便后续实现；
- 现阶段仍使用右 padding + debug 日志定位问题，等确认需求后再动手改。

