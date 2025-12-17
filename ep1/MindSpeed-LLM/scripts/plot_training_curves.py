#!/usr/bin/env python3
"""Plot training/validation loss curves from MindSpeed logs."""

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

# Regular expressions kept small and composable to make the parser resilient to
# formatting tweaks that frequently happen in long training logs.
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
    """Container for the parsed points of a single log file."""

    label: str
    tokens: List[float]
    train_loss: List[float]
    val_tokens: List[float]
    val_loss: List[float]
    seq_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "可视化 MindSpeed 训练日志中的 train/val loss，并把横轴换算成 token 数"
        )
    )
    parser.add_argument("logs", nargs="+", help="一个或多个待绘制的日志路径")
    parser.add_argument(
        "--output",
        default="outputs/training_curves.png",
        help="输出图像路径，默认 outputs/training_curves.png",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=50,
        help="train loss 平滑窗口大小 (step)，默认 50",
    )
    parser.add_argument(
        "--title",
        default="Training vs Validation Loss",
        help="图标题，默认 'Training vs Validation Loss'",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=None,
        help="如日志缺少 seq_length，可手动指定 batch seq length",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="保存图像的 DPI，默认 200",
    )
    return parser.parse_args()


def moving_average(values: Iterable[float], window: int) -> List[float]:
    """Return a causal moving average with adaptive window near the edges."""

    values = list(values)
    if window <= 1 or len(values) == 0:
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
            # 首先尝试解析 seq_length，以便后续能把 step 转成 token 数。
            if seq_length is None:
                seq_candidate = extract_seq_length(raw_line)
                if seq_candidate is not None:
                    seq_length = seq_candidate

            # 解析训练 iteration 行，需同时包含 lm loss 关键字。
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
                            f"在 {path} 中找不到 seq_length，请使用 --seq-length 指定"
                        )
                    if latest_gbs is None:
                        raise ValueError(
                            f"在 {path} 中找不到 global batch size 信息"
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
                if seq_length is None:
                    raise ValueError(
                        f"在 {path} 中找不到 seq_length，请使用 --seq-length 指定"
                    )
                if latest_gbs is None:
                    raise ValueError(
                        f"在 {path} 中找不到 global batch size 信息"
                    )
                token_x = val_iter * latest_gbs * seq_length
                val_tokens.append(token_x)
                val_loss.append(val_value)

    if seq_length is None:
        raise ValueError(f"{path} 既没有 seq_length 记录，也未通过 --seq-length 指定")
    if not tokens:
        raise ValueError(f"{path} 中没有解析到训练 loss 行")

    return CurveData(
        label=label,
        tokens=tokens,
        train_loss=train_loss,
        val_tokens=val_tokens,
        val_loss=val_loss,
        seq_length=seq_length,
    )


def plot_curves(curves: List[CurveData], args: argparse.Namespace) -> Path:
    plt.figure(figsize=(10, 6))
    color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    for curve in curves:
        color = next(color_cycle)
        smooth = moving_average(curve.train_loss, args.smooth_window)
        label_base = curve.label

        # 先画平滑 train loss (实线)，再叠加原始曲线 (半透明)，方便区分。
        plt.plot(
            curve.tokens,
            smooth,
            label=f"{label_base} train (smooth)",
            color=color,
            linestyle="-",
        )
        plt.plot(
            curve.tokens,
            curve.train_loss,
            label=f"{label_base} train (raw)",
            color=color,
            linestyle="-",
            alpha=0.5,
            linewidth=1.0,
        )

        if curve.val_tokens and curve.val_loss:
            plt.plot(
                curve.val_tokens,
                curve.val_loss,
                label=f"{label_base} val",
                color=color,
                linestyle="--",
                marker="o",
                linewidth=1.2,
            )

    plt.xlabel("Tokens seen")
    plt.ylabel("Loss")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 刻度格式化成更易读的单位（如 1.2B tokens）。
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

    curves: List[CurveData] = []
    for log_path in log_paths:
        if not log_path.is_file():
            raise FileNotFoundError(f"日志 {log_path} 不存在")
        curves.append(parse_log(log_path, args.seq_length))
        print(
            f"已解析 {log_path} ，step 数 {len(curves[-1].tokens)}, "
            f"val 点 {len(curves[-1].val_tokens)}"
        )

    output_path = plot_curves(curves, args)
    print(f"已保存图像到 {output_path}")


if __name__ == "__main__":
    main()
