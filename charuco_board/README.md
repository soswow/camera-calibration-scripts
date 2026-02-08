# ChArUco Board Generator

Generate print-ready ChArUco boards as single-page PDFs or tiled multi-page PDFs with crop marks, tile labels, and a minimap for assembly.

<table>
  <tr>
    <td><img src="../../docs/images/generate_charuco/charuco_pdf_result.jpg" alt="ChArUco PDF output" width="360"></td>
    <td><img src="../../docs/images/generate_charuco/charuco_A1_7x10_79p14mm_margin10mm_tileA3_2x2tiles_minimap.png" alt="Tiled output minimap" width="360"></td>
  </tr>
</table>

## Usage

Defaults:
- 10 x 7 squares
- 70 mm square size
- 0.7 marker proportion (49 mm markers)
- 300 DPI render
- PDF output by default

```bash
python scripts/charuco_board/generate_charuco.py \
  --paper A1 \
  --tile-paper A3 \
  --margin 10 \
  --squares-x 7 \
  --squares-y 10
```

You can use `--paper` together with `--squares-x/--squares-y` to have the square
size computed to fit the paper (respecting margins).

Example for a paper-sized board (A3 @ 300 DPI):

```bash
python scripts/charuco_board/generate_charuco.py \
  --paper A3 \
  --square-size 70 \
  --marker-proportion 0.7 \
  --output charuco_A3.pdf
```

## Tiled output (multi-page PDF)

```bash
python scripts/charuco_board/generate_charuco.py \
  --paper A1 \
  --tile-paper A3 \
  --margin 10 \
  --squares-x 8 \
  --squares-y 12
```

Tiling notes:
- `--tile-paper` requires `--paper` (the main board size).
- `--margin` applies per tile page.
- `--tile-bleed` controls the overflow beyond crop marks (default: 2mm).
- `--crop-mark` controls crop mark length (default: 5mm).
- A minimap PNG is written next to the PDF with `_minimap.png` suffix.

## Printing notes

- Laser printer preferred, matte paper
- 300 DPI or higher
- Disable any printer scaling (no “fit to page”)
- Print a reference ruler and verify scale
- After printing, measure square size; target error < 0.5 mm
- Mount to a flat surface; do not laminate
