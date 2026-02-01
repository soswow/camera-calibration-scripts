#!/usr/bin/env python3
"""Batch undistort images using calibration.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undistort a folder of images using calibration.json."
    )
    parser.add_argument(
        "--calibration",
        default="calibration.json",
        help="Path to calibration JSON. Default: calibration.json",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder or glob pattern for images.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for undistorted images.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Free scaling parameter for getOptimalNewCameraMatrix (0=crop, 1=keep all). Default: 1.0",
    )
    parser.add_argument(
        "--suffix",
        default="_undistorted",
        help="Suffix to append to output filenames. Default: _undistorted",
    )
    return parser.parse_args()


def _iter_images(input_arg: str) -> list[Path]:
    path = Path(input_arg)
    if any(ch in input_arg for ch in ["*", "?", "["]):
        return sorted(Path().glob(input_arg))
    if path.is_dir():
        return sorted(
            list(path.glob("*.png"))
            + list(path.glob("*.PNG"))
            + list(path.glob("*.jpg"))
            + list(path.glob("*.JPG"))
            + list(path.glob("*.jpeg"))
            + list(path.glob("*.JPEG"))
        )
    raise SystemExit(f"Input path not found: {input_arg}")


def main() -> int:
    args = _parse_args()
    with Path(args.calibration).open("r", encoding="utf-8") as handle:
        calib = json.load(handle)

    K = np.array(calib["camera_matrix"], dtype=np.float64)
    dist = np.array(calib["dist_coeffs"], dtype=np.float64)
    image_size = calib.get("image_size")

    images = _iter_images(args.input)
    if not images:
        raise SystemExit("No images found.")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    total = len(images)
    for idx, path in enumerate(images, start=1):
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if image_size and [w, h] != image_size:
            print(f"Skip {path.name}: size {w}x{h} != calibration {image_size[0]}x{image_size[1]}")
            continue
        newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), args.alpha, (w, h))
        undist = cv2.undistort(img, K, dist, None, newK)
        out_name = f"{path.stem}{args.suffix}{path.suffix}"
        out_path = out_dir / out_name
        ok = cv2.imwrite(str(out_path), undist)
        if ok:
            processed += 1
        if idx % 25 == 0 or idx == total:
            print(f"[{idx}/{total}] processed (written {processed})")

    print(f"Wrote {processed} undistorted images to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
