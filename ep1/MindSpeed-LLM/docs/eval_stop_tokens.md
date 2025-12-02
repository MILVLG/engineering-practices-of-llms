# Evaluation Stop Token Notes

## 背景
评测脚本基于 Megatron 的 `LLMChat.generate()`，默认只在生成到 `args.eos_id`（即 tokenizer 的 `eos_token_id`）时停机。在调试 MMLU 时，需要让模型遇到换行 `"\n"` 或 `"\n\n"` 即结束当前样本，因此对 stop token 机制做了如下梳理和改动：

## 原理回顾
- `LLMChat.generate()` 会读取 `stop_ids`，每轮生成后在 `_is_done()` 中判断 `prev == stop_id` 且该样本已经进入生成阶段（`started=True`）时立即停止。
- `started=True` 表示当前 `context_length` 已超过该样本原本的 prompt 长度；prompt 内的换行不会触发停止。
- 之前 `evaluation.py` 并未暴露 `--add-eos-token`，也默认按 token 字符串查 vocab，因此传入 `"\n"` 会被识别成 unk；必须用 `tokenizer.encode(token, add_special_tokens=False)` 来获得真实的单 token id。

## 代码更新
1. **`modellink/tasks/evaluation/utils.py`**
   - 新增 `--add-eos-token` 参数（`action='append'`/nargs）传递到 `args`。

2. **`evaluation.py`**
   - 读取 `args.add_eos_token`，在初始化 `LLMChat` 前保存。

3. **`LLMChat.generate()`**
   - 解析 `add_eos_token` 时使用 `tokenizer.encode(token, add_special_tokens=False)` 获取单 token id，并输出调试日志 `extra stop_ids=[...]`。

4. **`_sample_and_synchronize()` / `_post_process()`**
   - 如果开启 `debug_print_eval_batch`，日志中记录 `started_flags` 与 stop id 匹配情况，方便确认 stop 条件触发与否。

## 脚本配置
- `scripts/run_all_evals_exp.sh` 调用方式示例：
  ```bash
  --add-eos-token $'\n' $'\n\n'
  ```
  其中 `$'..'` 语法用于向 bash 传递真实的换行字符；多个停止 token 需要放在同一次 `--add-eos-token` 调用内。
- 如果使用常规脚本，可在命令行追加 `--add-eos-token` 参数；例如：
  ```bash
  DEBUG_MODE=1 ASCEND_VISIBLE_DEVICES=2,3 bash scripts/run_all_evals_exp.sh \
      outputs/zen_500m_ptd_24k \
      --add-eos-token $'\n' $'\n\n'
  ```

## 调试经验
- 观察日志时要确认 `texts=['\n'] started_flags=[True]` 以及后续的 `stop_id ... triggered...`，这意味着模型在生成阶段遇到 stop token 并提前结束。
- 如果只看到 `texts=['\n'] started_flags=[False]`，说明当前换行还在 prompt 中；继续等待模型进入输出阶段即可。
- 若加入多个 stop token 后发现 stop_ids 只有一个，检查脚本是否多次传入 `--add-eos-token`（后一次覆盖前一次）。

## 后续
- 目前 stop token 匹配只支持“单 token”；如需多 token 连续匹配（例如 "ABC"），需要拓展 `_post_process()` 跟踪最近 N 个 token。
