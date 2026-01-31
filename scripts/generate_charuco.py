#!/usr/bin/env python3
"""Generate a ChArUco board image for printing."""

from __future__ import annotations

import argparse
import sys

import cv2
from cv2 import aruco

DEFAULT_SQUARES_X = 10
DEFAULT_SQUARES_Y = 7
DEFAULT_SQUARE_SIZE_MM = 70.0
DEFAULT_MARKER_PROPORTION = 0.7
DEFAULT_DICTIONARY = "DICT_4X4_50"
DEFAULT_DPI = 300
DEFAULT_MARGIN_MM = 0.0
DEFAULT_OUTPUT = "auto"

PAPER_SIZES_MM = {
    "A0": (841.0, 1189.0),
    "A1": (594.0, 841.0),
    "A2": (420.0, 594.0),
    "A3": (297.0, 420.0),
    "A4": (210.0, 297.0),
    "LETTER": (216.0, 279.0),
    "LEGAL": (216.0, 356.0),
    "TABLOID": (279.0, 432.0),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ChArUco board image (OpenCV aruco).",
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
        help=f"Square side length in millimeters. Default: {DEFAULT_SQUARE_SIZE_MM}.",
    )
    parser.add_argument(
        "--marker-proportion",
        type=float,
        default=DEFAULT_MARKER_PROPORTION,
        help=(
            "Marker side length as a 0-1 proportion of square size. "
            f"Default: {DEFAULT_MARKER_PROPORTION}."
        ),
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY,
        help=f"OpenCV aruco dictionary name (e.g. DICT_4X4_50). Default: {DEFAULT_DICTIONARY}.",
    )
    parser.add_argument(
        "--paper",
        default=None,
        help="Paper size (e.g. A4, A3, Letter). Default: A4 when used.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Render DPI used to convert mm to pixels. Default: {DEFAULT_DPI}.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN_MM,
        help=f"Margin size in millimeters. Default: {DEFAULT_MARGIN_MM}.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output image path. Default: auto (constructed from arguments).",
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


def _render_board(board, size: tuple[int, int], margin_px: int, border_bits: int):
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=margin_px, borderBits=border_bits)
    return board.draw(size, marginSize=margin_px, borderBits=border_bits)


def _mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm / 25.4 * dpi))


def _paper_size_mm(name: str) -> tuple[float, float]:
    key = name.upper()
    if key not in PAPER_SIZES_MM:
        available = ", ".join(sorted(PAPER_SIZES_MM))
        raise SystemExit(f"Unknown paper size '{name}'. Available: {available}")
    return PAPER_SIZES_MM[key]


def _fmt_mm(mm: float) -> str:
    if abs(mm - round(mm)) < 1e-6:
        return str(int(round(mm)))
    return f"{mm:.2f}".rstrip("0").rstrip(".")


def _fmt_prop(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _sanitize_token(token: str) -> str:
    return token.replace(".", "p")


def _auto_output_name(
    *,
    args: argparse.Namespace,
    squares_x: int,
    squares_y: int,
    square_size: float,
    paper_label: str | None,
    provided_flags: set[str],
) -> str:
    parts: list[str] = ["charuco"]
    if paper_label:
        parts.append(paper_label)
    parts.append(f"{squares_x}x{squares_y}")

    extras: list[str] = []
    if (
        square_size != DEFAULT_SQUARE_SIZE_MM
        or "--square-size" in provided_flags
        or paper_label is not None
    ):
        extras.append(f"{_sanitize_token(_fmt_mm(square_size))}mm")
    if (
        args.marker_proportion != DEFAULT_MARKER_PROPORTION
        or "--marker-proportion" in provided_flags
    ):
        extras.append(f"m{_sanitize_token(_fmt_prop(args.marker_proportion))}")
    if args.dpi != DEFAULT_DPI or "--dpi" in provided_flags:
        extras.append(f"{args.dpi}dpi")
    if args.margin != DEFAULT_MARGIN_MM or "--margin" in provided_flags:
        extras.append(f"margin{_sanitize_token(_fmt_mm(args.margin))}mm")
    if args.dictionary != DEFAULT_DICTIONARY or "--dictionary" in provided_flags:
        extras.append(args.dictionary.lower())

    parts.extend(extras)
    return "_".join(parts) + ".png"


def main() -> int:
    args = _parse_args()
    argv = sys.argv[1:]
    provided_flags = set()
    for token in argv:
        if token.startswith("--"):
            provided_flags.add(token.split("=", 1)[0])
    paper_provided = "--paper" in argv
    squares_x_provided = "--squares-x" in argv
    squares_y_provided = "--squares-y" in argv
    squares_provided = squares_x_provided or squares_y_provided

    if paper_provided and squares_provided:
        raise SystemExit("Use either --paper or --squares-x/--squares-y, not both.")
    if squares_x_provided ^ squares_y_provided:
        raise SystemExit("Provide both --squares-x and --squares-y together.")

    if not (0.0 < args.marker_proportion < 1.0):
        raise SystemExit("marker-proportion must be in the 0-1 range (exclusive).")
    if args.square_size <= 0:
        raise SystemExit("square-size must be > 0.")
    if args.margin < 0:
        raise SystemExit("margin must be >= 0.")
    if args.dpi <= 0:
        raise SystemExit("dpi must be > 0.")

    if paper_provided:
        paper_width_mm, paper_height_mm = _paper_size_mm(args.paper or "A4")
        available_w = paper_width_mm - 2 * args.margin
        available_h = paper_height_mm - 2 * args.margin
        if available_w <= 0 or available_h <= 0:
            raise SystemExit("margin is too large for the selected paper size.")
        squares_x = int(round(available_w / args.square_size))
        squares_y = int(round(available_h / args.square_size))
        if squares_x < 2 or squares_y < 2:
            raise SystemExit("paper size is too small for the requested square-size.")
        actual_square_w = available_w / squares_x
        actual_square_h = available_h / squares_y
        square_size = min(actual_square_w, actual_square_h)
        output_width_mm = paper_width_mm
        output_height_mm = paper_height_mm
        paper_label = (args.paper or "A4").upper()
    else:
        squares_x = args.squares_x
        squares_y = args.squares_y
        if squares_x < 2 or squares_y < 2:
            raise SystemExit("squares-x and squares-y must be >= 2.")
        square_size = args.square_size
        output_width_mm = squares_x * square_size + 2 * args.margin
        output_height_mm = squares_y * square_size + 2 * args.margin
        paper_label = None

    marker_size = square_size * args.marker_proportion

    dictionary = _get_dictionary(args.dictionary)
    board = _create_board(
        squares_x,
        squares_y,
        square_size,
        marker_size,
        dictionary,
    )
    width_px = _mm_to_px(output_width_mm, args.dpi)
    height_px = _mm_to_px(output_height_mm, args.dpi)
    margin_px = _mm_to_px(args.margin, args.dpi)
    output_path = args.output
    if output_path == DEFAULT_OUTPUT:
        output_path = _auto_output_name(
            args=args,
            squares_x=squares_x,
            squares_y=squares_y,
            square_size=square_size,
            paper_label=paper_label,
            provided_flags=provided_flags,
        )
    img = _render_board(board, (width_px, height_px), margin_px, border_bits=1)
    ok = cv2.imwrite(output_path, img)
    if not ok:
        raise SystemExit(f"Failed to write output image: {output_path}")

    board_width_mm = squares_x * square_size
    board_height_mm = squares_y * square_size

    print("ChArUco board written:")
    print(f"  output: {output_path}")
    print(f"  squares: {squares_x} x {squares_y}")
    print(f"  square size (mm): {square_size}")
    if paper_provided and abs(square_size - args.square_size) > 0.01:
        print(f"  requested square size (mm): {args.square_size}")
    print(f"  marker proportion: {args.marker_proportion}")
    print(f"  marker size (mm): {marker_size}")
    print(f"  dictionary: {args.dictionary}")
    print(f"  board size (mm): {board_width_mm} x {board_height_mm}")
    if paper_provided:
        print(f"  paper: {paper_label}")
    print(f"  output size (mm): {output_width_mm} x {output_height_mm}")
    print(f"  pixels: {width_px} x {height_px}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
