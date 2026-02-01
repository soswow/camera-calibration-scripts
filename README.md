# Camera Calibration Helpers

Small set of scripts to support camera calibration tasks. First script: ChArUco board generator.

## Setup

Use your existing virtual env at `~/venvs/my`:

```bash
source ~/venvs/my/bin/activate
python -m pip install -r requirements.txt
```

## Generate a ChArUco board

The defaults match the PDF recommendations:
- 10 x 7 squares
- 70 mm square size
- 0.7 marker proportion (49 mm markers)
- 300 DPI render

```bash
python scripts/generate_charuco.py --output charuco_A4.png
```

Use either `--squares-x/--squares-y` or `--paper` (not both).

Example for a paper-sized board (A3 @ 300 DPI):

```bash
python scripts/generate_charuco.py \
  --paper A3 \
  --square-size 70 \
  --marker-proportion 0.7 \
  --output charuco_A3.png
```

Outputs (default directory: `output/`):
- Full board image (auto-named unless `--output` is set).

## Printing notes (from the PDF)

- Laser printer preferred, matte paper
- 300 DPI or higher
- Disable any printer scaling (no “fit to page”)
- Print a reference ruler and verify scale
- After printing, measure square size; target error < 0.5 mm
- Mount to a flat surface; do not laminate
