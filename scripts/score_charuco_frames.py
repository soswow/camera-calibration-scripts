#!/usr/bin/env python3
"""Score ChArUco video frames for sharpness and board detection quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from cv2 import aruco

DEFAULT_DICTIONARY = "DICT_4X4_50"
DEFAULT_SQUARES_X = 7
DEFAULT_SQUARES_Y = 10
DEFAULT_SQUARE_SIZE_MM = 79.14
DEFAULT_MARKER_PROPORTION = 0.7
DEFAULT_MIN_MARKERS = 1
DEFAULT_MIN_CHARUCO = 4
DEFAULT_WEIGHT_SHARPNESS = 0.7
DEFAULT_WEIGHT_CORNERS = 0.3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ChArUco frames and write a JSON list sorted by score."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder or glob pattern for PNG images.",
    )
    parser.add_argument(
        "--output",
        default="frame_scores.json",
        help="Output JSON path. Default: frame_scores.json",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY,
        help=f"aruco dictionary (e.g. DICT_4X4_50). Default: {DEFAULT_DICTIONARY}.",
    )
    parser.add_argument(
        "--squares-x",
        type=int,
        default=DEFAULT_SQUARES_X,
        help=f"Number of squares along X. Default: {DEFAULT_SQUARES_X}.",
    )
    parser.add_argument(
        "--squares-y",
        type=int,
        default=DEFAULT_SQUARES_Y,
        help=f"Number of squares along Y. Default: {DEFAULT_SQUARES_Y}.",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE_MM,
        help=f"Square side length in mm. Default: {DEFAULT_SQUARE_SIZE_MM}.",
    )
    parser.add_argument(
        "--marker-proportion",
        type=float,
        default=DEFAULT_MARKER_PROPORTION,
        help=f"Marker side length as a proportion of square size. Default: {DEFAULT_MARKER_PROPORTION}.",
    )
    parser.add_argument(
        "--min-markers",
        type=int,
        default=DEFAULT_MIN_MARKERS,
        help=f"Minimum markers required to score > 0. Default: {DEFAULT_MIN_MARKERS}.",
    )
    parser.add_argument(
        "--min-charuco",
        type=int,
        default=DEFAULT_MIN_CHARUCO,
        help=f"Minimum charuco corners required to score > 0. Default: {DEFAULT_MIN_CHARUCO}.",
    )
    parser.add_argument(
        "--weight-sharpness",
        type=float,
        default=DEFAULT_WEIGHT_SHARPNESS,
        help=f"Weight for sharpness score. Default: {DEFAULT_WEIGHT_SHARPNESS}.",
    )
    parser.add_argument(
        "--weight-corners",
        type=float,
        default=DEFAULT_WEIGHT_CORNERS,
        help=f"Weight for charuco corners score. Default: {DEFAULT_WEIGHT_CORNERS}.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N images. Default: 25 (set 0 to disable).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image debug info while processing.",
    )
    return parser.parse_args()


def _get_dictionary(name: str):
    if not hasattr(aruco, name):
        available = [k for k in dir(aruco) if k.startswith("DICT_")]
        raise SystemExit(
            f"Unknown dictionary '{name}'. Available examples: {', '.join(sorted(available)[:8])}"
        )
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def _create_board(
    squares_x: int,
    squares_y: int,
    square_size: float,
    marker_size: float,
    dictionary,
):
    if hasattr(aruco, "CharucoBoard"):
        return aruco.CharucoBoard(
            (squares_x, squares_y),
            square_size,
            marker_size,
            dictionary,
        )
    return aruco.CharucoBoard_create(
        squares_x,
        squares_y,
        square_size,
        marker_size,
        dictionary,
    )


def _iter_images(input_arg: str) -> list[Path]:
    path = Path(input_arg)
    if any(ch in input_arg for ch in ["*", "?", "["]):
        return sorted(Path().glob(input_arg))
    if path.is_dir():
        return sorted(list(path.glob("*.png")) + list(path.glob("*.PNG")))
    if path.is_file():
        return [path]
    raise SystemExit(f"Input path not found: {input_arg}")


def _detect_markers(gray, dictionary):
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
    return corners, ids


def _marker_bbox(corners, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not corners:
        return None
    pts = np.concatenate(corners, axis=0).reshape(-1, 2)
    x0 = int(np.floor(np.min(pts[:, 0])))
    y0 = int(np.floor(np.min(pts[:, 1])))
    x1 = int(np.ceil(np.max(pts[:, 0])))
    y1 = int(np.ceil(np.max(pts[:, 1])))
    pad = 5
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width - 1, x1 + pad)
    y1 = min(height - 1, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _proxy_metrics(
    bbox: tuple[int, int, int, int] | None, width: int, height: int
) -> dict:
    if bbox is None or width <= 0 or height <= 0:
        return {
            "bbox": None,
            "center_x": None,
            "center_y": None,
            "area_frac": None,
            "aspect": None,
        }
    x0, y0, x1, y1 = bbox
    w = max(1, x1 - x0 + 1)
    h = max(1, y1 - y0 + 1)
    center_x = (x0 + x1) / 2.0 / width
    center_y = (y0 + y1) / 2.0 / height
    area_frac = (w * h) / float(width * height)
    aspect = w / float(h)
    return {
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "center_x": float(center_x),
        "center_y": float(center_y),
        "area_frac": float(area_frac),
        "aspect": float(aspect),
    }


def _sharpness(gray, bbox: tuple[int, int, int, int] | None) -> float:
    if bbox:
        x0, y0, x1, y1 = bbox
        roi = gray[y0 : y1 + 1, x0 : x1 + 1]
    else:
        roi = gray
    if roi.size == 0:
        return 0.0
    lap = cv2.Laplacian(roi, cv2.CV_64F)
    return float(lap.var())


def _charuco_corners(gray, board, marker_corners, marker_ids) -> int:
    if marker_ids is None or len(marker_ids) == 0:
        return 0
    try:
        _, charuco_corners, _ = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
    except cv2.error:
        return 0
    if charuco_corners is None:
        return 0
    return int(len(charuco_corners))


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [0.0 for _ in values]
    scale = max_v - min_v
    return [(v - min_v) / scale for v in values]


def main() -> int:
    args = _parse_args()
    if args.squares_x < 2 or args.squares_y < 2:
        raise SystemExit("squares-x and squares-y must be >= 2.")
    if args.square_size <= 0:
        raise SystemExit("square-size must be > 0.")
    if not (0.0 < args.marker_proportion < 1.0):
        raise SystemExit("marker-proportion must be in the 0-1 range (exclusive).")
    if args.weight_sharpness < 0 or args.weight_corners < 0:
        raise SystemExit("weights must be >= 0.")
    if args.min_markers < 0 or args.min_charuco < 0:
        raise SystemExit("min thresholds must be >= 0.")

    images = _iter_images(args.input)
    if not images:
        raise SystemExit("No images found.")

    dictionary = _get_dictionary(args.dictionary)
    marker_size = args.square_size * args.marker_proportion
    board = _create_board(
        args.squares_x,
        args.squares_y,
        args.square_size,
        marker_size,
        dictionary,
    )

    records = []
    sharpness_vals = []
    corner_vals = []

    total = len(images)
    for idx, path in enumerate(images, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}: failed to read")
            continue
        height, width = img.shape[:2]
        corners, ids = _detect_markers(img, dictionary)
        num_markers = int(len(ids)) if ids is not None else 0
        bbox = _marker_bbox(corners, width, height)
        sharp_val = _sharpness(img, bbox)
        charuco_count = _charuco_corners(img, board, corners, ids)
        proxy = _proxy_metrics(bbox, width, height)

        if args.verbose:
            print(
                f"[{idx}/{total}] {path.name} "
                f"markers={num_markers} charuco={charuco_count} sharpness={sharp_val:.2f}"
            )
        elif args.progress_every and (idx % args.progress_every == 0 or idx == total):
            print(f"[{idx}/{total}] processed")

        records.append(
            {
                "file": str(path.resolve()),
                "sharpness": sharp_val,
                "charuco_corners": charuco_count,
                "markers": num_markers,
                **proxy,
            }
        )
        sharpness_vals.append(sharp_val)
        corner_vals.append(float(charuco_count))

    sharp_norm = _normalize(sharpness_vals)
    corner_norm = _normalize(corner_vals)
    total_weight = args.weight_sharpness + args.weight_corners
    if total_weight <= 0:
        total_weight = 1.0

    results = []
    for idx, rec in enumerate(records):
        score = (
            args.weight_sharpness * sharp_norm[idx]
            + args.weight_corners * corner_norm[idx]
        ) / total_weight
        if rec["markers"] < args.min_markers or rec["charuco_corners"] < args.min_charuco:
            score = 0.0
        results.append(
            {
                "file": rec["file"],
                "score": round(score, 6),
                "sharpness": round(float(rec["sharpness"]), 6),
                "charuco_corners": int(rec["charuco_corners"]),
                "markers": int(rec["markers"]),
                "bbox": rec["bbox"],
                "center_x": None if rec["center_x"] is None else round(rec["center_x"], 6),
                "center_y": None if rec["center_y"] is None else round(rec["center_y"], 6),
                "area_frac": None if rec["area_frac"] is None else round(rec["area_frac"], 6),
                "aspect": None if rec["aspect"] is None else round(rec["aspect"], 6),
            }
        )

    results.sort(key=lambda r: r["score"], reverse=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote {len(results)} scores to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
