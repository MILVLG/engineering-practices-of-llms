#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sample a smaller subset from Ultra-FineWeb-ShaoZW_subset.

The overall structure follows ``data_sample_example1.py`` but accepts an existing
dataset (already split into ``train`` / ``val``) and produces a new subset with:

* train samples taken from ``Ultra-FineWeb-ShaoZW_subset/train``;
* val split copied verbatim;
* target train storage size specified in GB (approximately 100 MB per parquet);
* controllable language ratio (zh/en) and score filtering.

During sampling the writer keeps a large in-memory buffer (~10x batch size),
randomly spills mini-batches to disk, and tracks disk usage in real time. This
avoids having to pre-compute sample counts and yields better shuffling for the
final shards.
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("content", pa.string()),
    ("score", pa.float64()),
    ("source", pa.string()),
    ("lang",   pa.string()),
])


class RollingParquetWriter:
    """Parquet writer that shuffles buffered rows before each flush."""

    def __init__(self, out_dir: Path, split: str, target_mb: int = 100,
                 compression: str = "snappy", rows_per_batch: int = 1000,
                 rng: Optional[random.Random] = None):
        self.base = out_dir / split
        self.base.mkdir(parents=True, exist_ok=True)
        self.target_bytes = max(1, target_mb) * 1024 * 1024
        self.compression = None if compression == "none" else compression
        self.rows_per_batch = rows_per_batch
        self.buffer_limit = rows_per_batch * 10
        self.rng = rng or random.Random()

        self.shard_idx = 0
        self.current_path: Optional[Path] = None
        self.writer: Optional[pq.ParquetWriter] = None
        self.buffer: List[Dict] = []

    def _next_path(self) -> Path:
        return self.base / f"part-{self.shard_idx:06d}.parquet"

    def _ensure_writer(self):
        if self.writer is None:
            self.current_path = self._next_path()
            self.writer = pq.ParquetWriter(
                self.current_path.as_posix(),
                SCHEMA,
                compression=self.compression,
                use_dictionary=True,
            )
            self.shard_idx += 1

    def _close_writer_only(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.current_path = None

    def _close_current(self):
        if self.writer is not None:
            self.flush()
            self._close_writer_only()

    def _current_size(self) -> int:
        if self.current_path and self.current_path.exists():
            return self.current_path.stat().st_size
        return 0

    def write_rows(self, rows: List[Dict]):
        self.buffer.extend(rows)
        while len(self.buffer) >= self.buffer_limit:
            self._emit_random_batch(self.rows_per_batch)

    def _emit_random_batch(self, batch_size: int):
        if not self.buffer:
            return
        take = min(batch_size, len(self.buffer))
        if take == len(self.buffer):
            batch = self.buffer
            self.buffer = []
        else:
            indices = sorted(self.rng.sample(range(len(self.buffer)), take), reverse=True)
            batch = [self.buffer.pop(i) for i in indices]
        self._ensure_writer()
        table = pa.Table.from_pylist(batch, schema=SCHEMA)
        self.writer.write_table(table)
        if self.current_path and self._current_size() >= self.target_bytes:
            self._close_writer_only()

    def finalize(self):
        self._close_current()

    def flush_partial(self):
        if len(self.buffer) >= self.rows_per_batch:
            self._emit_random_batch(self.rows_per_batch)

    def flush(self):
        if not self.buffer:
            return
        while self.buffer:
            batch_size = min(self.rows_per_batch, len(self.buffer))
            self._emit_random_batch(batch_size)

    def size_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.base.glob("*.parquet"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Sample subset from Ultra-FineWeb-ShaoZW_subset")
    p.add_argument("--input-root", type=str,
                   default="/docker/datasets/Ultra-FineWeb-ShaoZW_subset",
                   help="Existing Ultra-FineWeb-ShaoZW_subset 根目录")
    p.add_argument("--out-dir", type=str, default="./out_subset",
                   help="输出目录，将在其中创建 train/ 与 val/ 子目录")
    p.add_argument("--train-size-gb", type=float, required=True,
                   help="目标 train 数据集大小（GB）。用于估算采样规模")
    p.add_argument("--zh-ratio", type=float, default=0.5,
                   help="train 中中文样本占比 (0-1)")
    p.add_argument("--en-ratio", type=float, default=0.5,
                   help="train 中英文样本占比 (0-1)，两个比例需相加≈1")
    p.add_argument("--min-score-zh", type=float, default=0.0,
                   help="筛选中文样本的最低 score")
    p.add_argument("--min-score-en", type=float, default=0.0,
                   help="筛选英文样本的最低 score")
    p.add_argument("--target-shard-mb", type=int, default=100,
                   help="输出 train shard 的目标大小 (MB)")
    p.add_argument("--rows-per-batch", type=int, default=2000,
                   help="写 parquet 时的 batch 行数")
    p.add_argument("--seed", type=int, default=42, help="随机种子，用于遍历顺序")
    p.add_argument("--compression", type=str, default="snappy",
                   choices=["snappy", "zstd", "gzip", "brotli", "none"],
                   help="Parquet 压缩方式")
    return p.parse_args()


def validate_ratio(zh_ratio: float, en_ratio: float):
    if zh_ratio < 0 or en_ratio < 0:
        raise ValueError("language ratio must be >= 0")
    if zh_ratio + en_ratio == 0:
        raise ValueError("至少一个语言比例需要大于0")
    if abs((zh_ratio + en_ratio) - 1.0) > 1e-6:
        raise ValueError("当前脚本仅支持 zh + en = 1 的比例设定")


def list_parquet_files(root: Path, split: str) -> List[Path]:
    paths = sorted((root / split).glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"未找到 {split}/*.parquet")
    return paths


def approx_row_bytes(row: Dict) -> int:
    content = row.get("content") or ""
    source = row.get("source") or ""
    lang = row.get("lang") or ""
    bytes_est = (len(content.encode("utf-8")) +
                 len(source.encode("utf-8")) +
                 len(lang.encode("utf-8")) + 8)  # +8 for score(float64)
    return max(bytes_est, 1)


def iter_rows(paths: List[Path]) -> Iterator[Dict]:
    for p in paths:
        pf = pq.ParquetFile(p)
        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["content", "score", "source", "lang"])
            for row in table.to_pylist():
                yield row


def sample_train(paths: List[Path], total_bytes_target: float,
                 zh_ratio: float, en_ratio: float,
                 min_score_zh: float, min_score_en: float,
                 writer: RollingParquetWriter, rng: random.Random) -> Tuple[int, int, float, float]:
    if total_bytes_target <= 0:
        return 0, 0, 0.0, 0.0

    shuffled_paths = paths[:]
    rng.shuffle(shuffled_paths)

    zh_written = 0
    en_written = 0
    zh_bytes_acc = 0.0
    en_bytes_acc = 0.0
    approx_total_bytes = 0.0
    approx_total_bytes_target = real_total_bytes_target = total_bytes_target

    zh_bytes_target = total_bytes_target * zh_ratio
    en_bytes_target = total_bytes_target * en_ratio

    stop = False

    for ii, path in enumerate(shuffled_paths):
        if stop:
            break
        print(f"[info][{ii}] sampling {path.name} (zh_rows={zh_written}, en_rows={en_written}, ~bytes={approx_total_bytes/1e6:.2f}M), real disk used: {writer.size_on_disk()/1e6:.2f}M, approx_target: {approx_total_bytes_target/1e6:.2f}M")
        pf = pq.ParquetFile(path)
        row_group_indices = list(range(pf.num_row_groups))
        rng.shuffle(row_group_indices)
        for rg_idx in row_group_indices:
            if stop:
                break
            table = pf.read_row_group(rg_idx, columns=["content", "score", "source", "lang"])
            rows = table.to_pylist()
            rng.shuffle(rows)
            batch: List[Dict] = []
            for row in rows:
                lang = row.get("lang")
                if lang == "zh":
                    if zh_bytes_target == 0:
                        continue
                    score = row.get("score")
                    if score is None or float(score) < min_score_zh:
                        continue
                    if zh_bytes_acc >= zh_bytes_target and en_bytes_acc < en_bytes_target:
                        continue
                    row_bytes = approx_row_bytes(row)
                    batch.append({
                        "content": row.get("content", ""),
                        "score": float(score),
                        "source": row.get("source", ""),
                        "lang": "zh",
                    })
                    zh_written += 1
                    zh_bytes_acc += row_bytes
                    approx_total_bytes += row_bytes
                elif lang == "en":
                    if en_bytes_target == 0:
                        continue
                    score = row.get("score")
                    if score is None or float(score) < min_score_en:
                        continue
                    if en_bytes_acc >= en_bytes_target and zh_bytes_acc < zh_bytes_target:
                        continue
                    row_bytes = approx_row_bytes(row)
                    batch.append({
                        "content": row.get("content", ""),
                        "score": float(score),
                        "source": row.get("source", ""),
                        "lang": "en",
                    })
                    en_written += 1
                    en_bytes_acc += row_bytes
                    approx_total_bytes += row_bytes
                if approx_total_bytes >= approx_total_bytes_target:
                    stop = True
                    break
            if batch:
                writer.write_rows(batch)
                current_disk = writer.size_on_disk()
                if current_disk == 0:
                    writer.flush_partial()
                    current_disk = writer.size_on_disk()
                if current_disk >= real_total_bytes_target:
                    stop = True
                    break
                if current_disk > 0:
                    approx_total_bytes_target = real_total_bytes_target * approx_total_bytes / current_disk
                    en_bytes_target = approx_total_bytes_target * en_ratio
                    zh_bytes_target = approx_total_bytes_target * zh_ratio

    writer.flush()
    return zh_written, en_written, zh_bytes_acc, en_bytes_acc


def copy_val_split(src_root: Path, out_root: Path):
    src_val = src_root / "val"
    dst_val = out_root / "val"
    if dst_val.exists():
        shutil.rmtree(dst_val)
    dst_val.mkdir(parents=True, exist_ok=True)
    for item in src_val.iterdir():
        target = dst_val / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main():
    args = parse_args()
    validate_ratio(args.zh_ratio, args.en_ratio)

    rng = random.Random(args.seed)

    input_root = Path(args.input_root).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    train_paths = list_parquet_files(input_root, "train")
    val_paths = list_parquet_files(input_root, "val")
    print(f"Found {len(train_paths)} train parquet files.")
    print(f"Found {len(val_paths)} val parquet files.")

    # ---- 复制 val ----
    copy_val_split(input_root, out_root)

    total_bytes_target = max(args.train_size_gb, 0.01) * (1024 ** 3)
    zh_bytes_target = total_bytes_target * args.zh_ratio
    en_bytes_target = total_bytes_target * args.en_ratio
    num_train_shards = max(1, math.ceil((total_bytes_target / (1024 ** 2)) / args.target_shard_mb))
    print(f"目标 train size ≈ {args.train_size_gb} GB，预计写入 {num_train_shards} 个 parquet (约 {args.target_shard_mb} MB/个)")
    print(f"中文目标字节 ≈ {zh_bytes_target/1e6:.2f} MB，英文目标字节 ≈ {en_bytes_target/1e6:.2f} MB")

    train_writer = RollingParquetWriter(out_root, "train",
                                        target_mb=args.target_shard_mb,
                                        compression=args.compression,
                                        rows_per_batch=args.rows_per_batch,
                                        rng=rng)

    zh_written, en_written, zh_bytes_acc, en_bytes_acc = sample_train(
        train_paths,
        total_bytes_target,
        args.zh_ratio,
        args.en_ratio,
        args.min_score_zh,
        args.min_score_en,
        train_writer,
        rng,
    )

    approx_total_bytes = zh_bytes_acc + en_bytes_acc
    if zh_bytes_target > 0 and zh_bytes_acc < zh_bytes_target:
        print(f"[warn] 中文样本仅采集约 {zh_bytes_acc/1e6:.2f} MB / {zh_bytes_target/1e6:.2f} MB，可能因源数据不足或过滤条件过严。")
    if en_bytes_target > 0 and en_bytes_acc < en_bytes_target:
        print(f"[warn] 英文样本仅采集约 {en_bytes_acc/1e6:.2f} MB / {en_bytes_target/1e6:.2f} MB，可能因源数据不足或过滤条件过严。")

    train_writer.finalize()

    train_files = sorted((out_root / "train").glob("*.parquet"))
    actual_total_bytes = sum(f.stat().st_size for f in train_files)
    actual_total_gb = actual_total_bytes / (1024 ** 3)
    actual_num_shards = len(train_files)

    meta_path = out_root / "META_SAMPLE.txt"
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(
            "subset: data_sample_example2\n"
            f"source_root: {input_root}\n"
            f"train_size_gb_target: {args.train_size_gb}\n"
            f"target_shard_mb: {args.target_shard_mb}\n"
            f"estimated_train_shards: {num_train_shards}\n"
            f"actual_train_shards: {actual_num_shards}\n"
            f"actual_train_size_gb: {actual_total_gb:.4f}\n"
            f"zh_ratio: {args.zh_ratio}, en_ratio: {args.en_ratio}\n"
            f"min_score_zh: {args.min_score_zh}, min_score_en: {args.min_score_en}\n"
            f"rows_written_zh: {zh_written}\n"
            f"rows_written_en: {en_written}\n"
            f"approx_bytes_zh: {zh_bytes_acc:.0f}\n"
            f"approx_bytes_en: {en_bytes_acc:.0f}\n"
            f"approx_total_bytes: {approx_total_bytes:.0f}\n"
            "notes: 尺寸基于平均估算，实际大小可能略有偏差。\n"
        )

    print("采样完成。输出目录:", out_root.as_posix())
    print(f"实际 train 文件 {actual_num_shards} 个，约 {actual_total_gb:.3f} GB")

    # ---- 最后对 train 分片做额外 shuffle 操作 ----
    # 无需额外打散：RollingParquetWriter 已在写入阶段随机抽取 batch



if __name__ == "__main__":
    main()
