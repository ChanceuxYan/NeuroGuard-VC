#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 LibriSpeech 目录下的 train-clean-360 / test-clean / dev-clean 转换为 CSV

默认输入根目录: /work/2023/yanjunzhe/LibriSpeech_wav/
默认输出目录: /home/yanjunzhe/project/WM-V2/NeuroGuard-VC/data

生成的 CSV 形式与项目现有的 train/val CSV 相同:
    wav_path,speaker_id

用法:
    python scripts/build_librispeech_csv.py \
        --root /work/2023/yanjunzhe/LibriSpeech_wav \
        --out_dir /home/yanjunzhe/project/WM-V2/NeuroGuard-VC/data
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def collect_subset(subset_dir: Path) -> List[Tuple[str, str]]:
    """遍历子集目录，返回 (wav_path, speaker_id) 列表"""
    rows: List[Tuple[str, str]] = []
    for wav_path in subset_dir.rglob("*.wav"):
        # LibriSpeech 目录结构: subset/speaker_id/chapter_id/xxxx.wav
        try:
            speaker_id = wav_path.parents[1].name  # 上两级目录为 speaker_id
        except IndexError:
            # 结构异常时跳过
            continue
        rows.append((str(wav_path), speaker_id))
    return rows


def write_csv(rows: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "speaker_id"])
        writer.writerows(rows)
    print(f"✓ Saved {len(rows)} rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Build CSVs for LibriSpeech subsets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/work/2023/yanjunzhe/LibriSpeech_wav"),
        help="LibriSpeech_wav 根目录，需包含 train-clean-360 / test-clean / dev-clean",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("/home/yanjunzhe/project/WM-V2/NeuroGuard-VC/data"),
        help="输出 CSV 存放目录",
    )
    args = parser.parse_args()

    subset_map = {
        "train-clean-360": "train.csv",
        "test-clean": "test.csv",
        "dev-clean": "dev.csv",
    }

    for subset, csv_name in subset_map.items():
        subset_dir = args.root / subset
        if not subset_dir.exists():
            print(f"⚠ 跳过: {subset_dir} 不存在")
            continue
        print(f"收集子集: {subset_dir}")
        rows = collect_subset(subset_dir)
        out_path = args.out_dir / csv_name
        write_csv(rows, out_path)


if __name__ == "__main__":
    main()

