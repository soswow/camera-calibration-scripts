#!/usr/bin/env python3
"""Select a diverse subset of frames based on proxy pose metrics."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path


DEFAULT_SELECT = 50
DEFAULT_TOP_K = 300


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a diverse subset of frames from a score JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSON produced by score_charuco_frames.py",
    )
    parser.add_argument(
        "--output",
        default="selected_frames.json",
        help="Output JSON path. Default: selected_frames.json",
    )
    parser.add_argument(
        "--copy-to",
        default=None,
        help="Optional directory to copy selected frames into.",
    )
    parser.add_argument(
        "--select",
        type=int,
        default=DEFAULT_SELECT,
        help=f"How many frames to select. Default: {DEFAULT_SELECT}.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Consider only the top-K by score before diversity selection. Default: {DEFAULT_TOP_K}. Set 0 to disable.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum score required to be considered. Default: 0.",
    )
    return parser.parse_args()


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [0.0 for _ in values]
    scale = max_v - min_v
    return [(v - min_v) / scale for v in values]


def _distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> int:
    args = _parse_args()
    if args.select <= 0:
        raise SystemExit("--select must be > 0.")
    if args.top_k < 0:
        raise SystemExit("--top-k must be >= 0.")

    input_path = Path(args.input)
    with input_path.open("r", encoding="utf-8") as handle:
        items = json.load(handle)

    candidates = []
    for item in items:
        score = float(item.get("score", 0.0))
        if score < args.min_score:
            continue
        center_x = item.get("center_x")
        center_y = item.get("center_y")
        area_frac = item.get("area_frac")
        aspect = item.get("aspect")
        if None in (center_x, center_y, area_frac, aspect):
            continue
        candidates.append(item)

    if not candidates:
        raise SystemExit("No candidates with complete proxy metrics.")

    candidates.sort(key=lambda r: float(r.get("score", 0.0)), reverse=True)
    if args.top_k and len(candidates) > args.top_k:
        candidates = candidates[: args.top_k]

    features = []
    for item in candidates:
        features.append(
            (
                float(item["center_x"]),
                float(item["center_y"]),
                float(item["area_frac"]),
                float(item["aspect"]),
            )
        )

    # Normalize per-dimension for fair distance.
    dim_values = list(zip(*features))
    norm_dims = [_normalize(list(dim)) for dim in dim_values]
    norm_features = list(zip(*norm_dims))

    selected_indices: list[int] = []
    selected = []

    # Start with highest score.
    selected_indices.append(0)
    selected.append(candidates[0])

    while len(selected_indices) < min(args.select, len(candidates)):
        best_idx = None
        best_dist = -1.0
        for idx, feat in enumerate(norm_features):
            if idx in selected_indices:
                continue
            min_dist = min(
                _distance(feat, norm_features[s_idx]) for s_idx in selected_indices
            )
            if min_dist > best_dist:
                best_dist = min_dist
                best_idx = idx
        if best_idx is None:
            break
        selected_indices.append(best_idx)
        selected.append(candidates[best_idx])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, indent=2)

    if args.copy_to:
        copy_dir = Path(args.copy_to)
        copy_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(selected, start=1):
            src = Path(item["file"])
            if not src.exists():
                continue
            dst = copy_dir / src.name
            if dst.exists():
                dst = copy_dir / f"{src.stem}_{idx}{src.suffix}"
            shutil.copy2(src, dst)

    print(f"Selected {len(selected)} frames -> {output_path}")
    if args.copy_to:
        print(f"Copied files to {copy_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
