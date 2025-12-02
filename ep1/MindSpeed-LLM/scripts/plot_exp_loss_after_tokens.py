#!/usr/bin/env python3
"""Plot exp(loss) curves after a token threshold from MindSpeed logs."""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 与 plot_training_curves.py 保持同样的解析逻辑，避免日志格式小变动导致脚本不可用。
ITERATION_RE = re.compile(r"iteration\s+(\d+)\s*/", re.IGNORECASE)
CONSUMED_RE = re.compile(r"consumed samples:\s*(\d+)", re.IGNORECASE)
GLOBAL_BATCH_RE = re.compile(r"global batch size:\s*(\d+)", re.IGNORECASE)
TRAIN_LOSS_RE = re.compile(r"lm loss:\s*([0-9.+\-Ee]+)", re.IGNORECASE)
VAL_LINE_RE = re.compile(
    r"validation loss at iteration\s+(\d+)\s*\|[^|]*lm loss value:\s*([0-9.+\-Ee]+)",
    re.IGNORECASE,
)
SEQ_PATTERNS = [
    re.compile(r"\bseq_length\b.*?(\d+)", re.IGNORECASE),
    re.compile(r"\bencoder_seq_length\b.*?(\d+)", re.IGNORECASE),
    re.compile(r"\bbatch_seq_len\b.*?(\d+)", re.IGNORECASE),
]


@dataclass
class CurveData:
    label: str
    tokens: List[float]
    train_loss: List[float]
    val_tokens: List[float]
    val_loss: List[float]
    seq_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="绘制 MindSpeed 日志中 5B token 之后的 exp(loss) 曲线"
    )
    parser.add_argument("logs", nargs="+", help="一个或多个日志路径")
    parser.add_argument(
        "--output",
        default="outputs/exp_loss_after_5B.png",
        help="输出图像路径，默认 outputs/exp_loss_after_5B.png",
    )
    parser.add_argument(
        "--token-threshold",
        type=float,
        default=5e9,
        help="起始 token 数阈值，默认 5e9",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="train loss 平滑窗口，默认 50",
    )
    parser.add_argument(
        "--title",
        default="exp(loss) after threshold",
        help="图标题，默认 'exp(loss) after threshold'",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="若日志中缺少 seq_length，可通过此参数指定",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="保存图像时的 DPI，默认 200",
    )
    return parser.parse_args()


def moving_average(values: Iterable[float], window: int) -> List[float]:
    values = list(values)
    if window <= 1 or not values:
        return values

    window = min(window, len(values))
    prefix = [0.0]
    for val in values:
        prefix.append(prefix[-1] + val)

    smoothed: List[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        count = idx - start + 1
        avg = (prefix[idx + 1] - prefix[start]) / count
        smoothed.append(avg)
    return smoothed


def extract_seq_length(line: str) -> Optional[int]:
    for pattern in SEQ_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def parse_log(path: Path, fallback_seq_len: Optional[int]) -> CurveData:
    label = path.stem
    seq_length = fallback_seq_len
    tokens: List[float] = []
    train_loss: List[float] = []
    val_tokens: List[float] = []
    val_loss: List[float] = []
    latest_gbs: Optional[int] = None

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            if seq_length is None:
                seq_candidate = extract_seq_length(raw_line)
                if seq_candidate is not None:
                    seq_length = seq_candidate

            if "iteration" in raw_line and "lm loss" in raw_line:
                iter_match = ITERATION_RE.search(raw_line)
                loss_match = TRAIN_LOSS_RE.search(raw_line)
                consumed_match = CONSUMED_RE.search(raw_line)
                gbs_match = GLOBAL_BATCH_RE.search(raw_line)
                if iter_match and loss_match:
                    iteration = int(iter_match.group(1))
                    loss_value = float(loss_match.group(1))
                    if gbs_match:
                        latest_gbs = int(gbs_match.group(1))
                    if seq_length is None:
                        raise ValueError(
                            f"在 {path} 中找不到 seq_length，需传入 --seq-length"
                        )
                    if latest_gbs is None:
                        raise ValueError(
                            f"在 {path} 中缺少 global batch size 信息"
                        )
                    if consumed_match:
                        consumed = int(consumed_match.group(1))
                        token_x = consumed * seq_length
                    else:
                        token_x = iteration * latest_gbs * seq_length
                    tokens.append(token_x)
                    train_loss.append(loss_value)

            val_match = VAL_LINE_RE.search(raw_line)
            if val_match:
                val_iter = int(val_match.group(1))
                val_value = float(val_match.group(2))
                if seq_length is None or latest_gbs is None:
                    raise ValueError(
                        f"{path} 中解析 val loss 需要 seq_length 和 global batch size"
                    )
                token_x = val_iter * latest_gbs * seq_length
                val_tokens.append(token_x)
                val_loss.append(val_value)

    if seq_length is None:
        raise ValueError(f"{path} 既未记录 seq_length，也未通过 --seq-length 指定")
    if not tokens:
        raise ValueError(f"{path} 中未能解析到训练 loss")

    return CurveData(
        label=label,
        tokens=tokens,
        train_loss=train_loss,
        val_tokens=val_tokens,
        val_loss=val_loss,
        seq_length=seq_length,
    )


def filter_after_threshold(curve: CurveData, threshold: float) -> CurveData:
    def filter_pairs(xs: List[float], ys: List[float]) -> tuple[List[float], List[float]]:
        filtered_x: List[float] = []
        filtered_y: List[float] = []
        for x, y in zip(xs, ys):
            if x >= threshold:
                filtered_x.append(x)
                filtered_y.append(y)
        return filtered_x, filtered_y

    train_x, train_y = filter_pairs(curve.tokens, curve.train_loss)
    val_x, val_y = filter_pairs(curve.val_tokens, curve.val_loss)

    return CurveData(
        label=curve.label,
        tokens=train_x,
        train_loss=train_y,
        val_tokens=val_x,
        val_loss=val_y,
        seq_length=curve.seq_length,
    )


def plot_curves(curves: List[CurveData], args: argparse.Namespace) -> Path:
    plt.figure(figsize=(10, 6))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    any_plotted = False

    for curve in curves:
        if not curve.tokens:
            print(f"日志 {curve.label} 在 {args.token_threshold:.2e} tokens 之后没有 train 数据")
            continue
        color = next(color_cycle)
        smooth_loss = moving_average(curve.train_loss, args.smooth_window)
        smooth_ppl = [math.exp(v) for v in smooth_loss]
        raw_ppl = [math.exp(v) for v in curve.train_loss]

        plt.plot(
            curve.tokens,
            smooth_ppl,
            label=f"{curve.label} train exp(loss)",
            color=color,
            linestyle="-",
        )
        plt.plot(
            curve.tokens,
            raw_ppl,
            label=f"{curve.label} train raw",  # 原始曲线半透明方便参考
            color=color,
            linestyle=":",
            alpha=0.5,
        )

        if curve.val_tokens:
            val_ppl = [math.exp(v) for v in curve.val_loss]
            plt.plot(
                curve.val_tokens,
                val_ppl,
                label=f"{curve.label} val exp(loss)",
                color=color,
                linestyle="--",
                marker="o",
            )
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("所有日志在给定阈值之后都没有数据，无法绘图")

    plt.xlabel("Tokens seen")
    plt.ylabel("exp(loss) (≈ perplexity)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    def format_tokens(value: float, _: int) -> str:
        if value == 0:
            return "0"
        units = [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]
        for threshold, suffix in units:
            if value >= threshold:
                return f"{value / threshold:.2f}{suffix}"
        return f"{value:.0f}"

    ax = plt.gca()
    ax.xaxis.set_major_formatter(FuncFormatter(format_tokens))
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=args.dpi)
    return output_path


def main() -> None:
    args = parse_args()
    log_paths = [Path(p) for p in args.logs]

    parsed: List[CurveData] = []
    for log_path in log_paths:
        if not log_path.is_file():
            raise FileNotFoundError(f"日志 {log_path} 不存在")
        curve = parse_log(log_path, args.seq_length)
        filtered = filter_after_threshold(curve, args.token_threshold)
        parsed.append(filtered)
        print(
            f"日志 {log_path}：总 step {len(curve.tokens)}，阈值后点数 {len(filtered.tokens)}"
        )

    output_path = plot_curves(parsed, args)
    print(
        f"已根据 token 阈值 {args.token_threshold:.2e} 绘制 exp(loss) 曲线：{output_path}"
    )


if __name__ == "__main__":
    main()
