#!/usr/bin/env python3
"""Calibrate camera intrinsics from ChArUco frames."""

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
DEFAULT_MIN_CHARUCO = 4
DEFAULT_MIN_MARKERS = 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate camera intrinsics using ChArUco frames."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input folder, glob pattern, or JSON file with frame list.",
    )
    parser.add_argument(
        "--output",
        default="calibration.json",
        help="Output JSON path. Default: calibration.json",
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
        help=f"Minimum markers required for a frame. Default: {DEFAULT_MIN_MARKERS}.",
    )
    parser.add_argument(
        "--min-charuco",
        type=int,
        default=DEFAULT_MIN_CHARUCO,
        help=f"Minimum ChArUco corners required for a frame. Default: {DEFAULT_MIN_CHARUCO}.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N images. Default: 25 (set 0 to disable).",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.0,
        help=(
            "Remove frames with per-view error above this threshold (pixels) and "
            "update --prune-json. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--prune-json",
        default=None,
        help="JSON file to update when pruning outliers. Default: input JSON (if provided).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image debug info.",
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
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        files = []
        for item in items:
            file_path = item.get("file")
            if file_path:
                files.append(Path(file_path))
        return files
    if any(ch in input_arg for ch in ["*", "?", "["]):
        return sorted(Path().glob(input_arg))
    if path.is_dir():
        return sorted(list(path.glob("*.png")) + list(path.glob("*.PNG")))
    if path.is_file():
        raise SystemExit("Input must be a folder, glob pattern, or JSON file.")
    raise SystemExit(f"Input path not found: {input_arg}")


def _detect_markers(gray, dictionary):
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary)
    return corners, ids


def _homography_ok(board, charuco_corners, charuco_ids) -> bool:
    if charuco_ids is None or charuco_corners is None:
        return False
    ids = charuco_ids.flatten()
    if len(ids) < 4:
        return False
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    obj_pts = corners[ids][:, :2].astype(np.float32)
    img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
    if len(obj_pts) < 4 or len(img_pts) < 4:
        return False
    homography, _ = cv2.findHomography(obj_pts, img_pts, 0)
    return homography is not None


def _compute_per_view_errors(
    board, all_corners, all_ids, rvecs, tvecs, camera_matrix, dist_coeffs
) -> list[float]:
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    per_view = []
    for idx, (charuco_corners, charuco_ids) in enumerate(zip(all_corners, all_ids)):
        ids = charuco_ids.flatten()
        obj_pts = corners[ids].astype(np.float32)
        img_pts = charuco_corners.reshape(-1, 2).astype(np.float32)
        if len(obj_pts) < 4:
            per_view.append(float("inf"))
            continue
        proj, _ = cv2.projectPoints(
            obj_pts, rvecs[idx], tvecs[idx], camera_matrix, dist_coeffs
        )
        proj = proj.reshape(-1, 2)
        diff = img_pts - proj
        err = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
        per_view.append(float(err))
    return per_view


def main() -> int:
    args = _parse_args()
    if args.squares_x < 2 or args.squares_y < 2:
        raise SystemExit("squares-x and squares-y must be >= 2.")
    if args.square_size <= 0:
        raise SystemExit("square-size must be > 0.")
    if not (0.0 < args.marker_proportion < 1.0):
        raise SystemExit("marker-proportion must be in the 0-1 range (exclusive).")
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

    all_corners = []
    all_ids = []
    used_files = []
    image_size = None

    total = len(images)
    for idx, path in enumerate(images, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}: failed to read")
            continue
        if image_size is None:
            image_size = (img.shape[1], img.shape[0])

        corners, ids = _detect_markers(img, dictionary)
        num_markers = int(len(ids)) if ids is not None else 0
        if num_markers < args.min_markers:
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}: markers={num_markers} (skip)")
            continue
        try:
            retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                corners, ids, img, board
            )
        except cv2.error:
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}: interpolate failed")
            continue
        if charuco_corners is None or len(charuco_corners) < args.min_charuco:
            if args.verbose:
                count = 0 if charuco_corners is None else len(charuco_corners)
                print(f"[{idx}/{total}] {path.name}: charuco={count} (skip)")
            continue
        if not _homography_ok(board, charuco_corners, charuco_ids):
            if args.verbose:
                print(f"[{idx}/{total}] {path.name}: homography failed (skip)")
            continue

        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        used_files.append(str(path.resolve()))

        if args.verbose:
            print(
                f"[{idx}/{total}] {path.name}: markers={num_markers} "
                f"charuco={len(charuco_corners)} (use)"
            )
        elif args.progress_every and (idx % args.progress_every == 0 or idx == total):
            print(f"[{idx}/{total}] processed")

    if not all_corners:
        raise SystemExit("No valid frames with enough ChArUco corners.")

    if image_size is None:
        raise SystemExit("Unable to determine image size.")

    calibration_method = "calibrateCameraCharuco"
    std_intrinsics = None
    std_extrinsics = None
    per_view_errors = None

    if hasattr(aruco, "calibrateCameraCharucoExtended"):
        try:
            (
                ret,
                camera_matrix,
                dist_coeffs,
                rvecs,
                tvecs,
                std_intrinsics,
                std_extrinsics,
                per_view_errors,
            ) = aruco.calibrateCameraCharucoExtended(
                all_corners,
                all_ids,
                board,
                image_size,
                None,
                None,
            )
            calibration_method = "calibrateCameraCharucoExtended"
        except cv2.error:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                all_corners,
                all_ids,
                board,
                image_size,
                None,
                None,
            )
    else:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
            all_corners,
            all_ids,
            board,
            image_size,
            None,
            None,
        )

    result = {
        "image_size": [int(image_size[0]), int(image_size[1])],
        "board": {
            "squares_x": int(args.squares_x),
            "squares_y": int(args.squares_y),
            "square_size_mm": float(args.square_size),
            "marker_size_mm": float(marker_size),
            "dictionary": args.dictionary,
        },
        "frames_used": len(used_files),
        "reprojection_error": float(ret),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "used_files": used_files,
        "calibration_method": calibration_method,
    }
    if per_view_errors is None:
        per_view_errors = _compute_per_view_errors(
            board, all_corners, all_ids, rvecs, tvecs, camera_matrix, dist_coeffs
        )
    if per_view_errors is not None:
        result["per_view_errors"] = (
            per_view_errors.tolist()
            if hasattr(per_view_errors, "tolist")
            else list(per_view_errors)
        )
    if std_intrinsics is not None:
        result["std_intrinsics"] = std_intrinsics.tolist()
    if std_extrinsics is not None:
        result["std_extrinsics"] = std_extrinsics.tolist()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(f"Wrote calibration to {output_path}")
    print(f"Frames used: {len(used_files)}")
    if args.prune_threshold > 0:
        if per_view_errors is None:
            raise SystemExit("Per-view errors are unavailable; cannot prune.")
        remove_files = {
            used_files[i]
            for i, err in enumerate(per_view_errors)
            if err > args.prune_threshold
        }
        if not remove_files:
            print("No frames exceeded prune threshold.")
            return 0
        prune_path = Path(args.prune_json) if args.prune_json else None
        if prune_path is None:
            input_path = Path(args.input)
            if input_path.suffix.lower() != ".json":
                raise SystemExit(
                    "Prune requires --prune-json when input is not a JSON file."
                )
            prune_path = input_path
        if not prune_path.exists():
            raise SystemExit(f"Prune JSON not found: {prune_path}")
        with prune_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        kept = []
        removed = 0
        for item in items:
            file_path = item.get("file")
            if not file_path:
                kept.append(item)
                continue
            try:
                resolved = str(Path(file_path).resolve())
            except OSError:
                resolved = file_path
            if resolved in remove_files:
                removed += 1
                continue
            kept.append(item)
        with prune_path.open("w", encoding="utf-8") as handle:
            json.dump(kept, handle, indent=2)
        print(f"Pruned {removed} frames from {prune_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
