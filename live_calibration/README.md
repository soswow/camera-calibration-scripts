# Live ChArUco Calibration

Interactive capture tool for collecting ChArUco observations in real time. It shows a live preview with an optional coverage heatmap, accepts frames that meet your detection/quality thresholds, and periodically recalibrates to give immediate feedback on reprojection error and coverage gaps.

See demo here: https://www.instagram.com/p/DUfEMg_k39c/

<img src="../docs/images/live_charuco_calibration/demo-screenshot.jpg" alt="Live ChArUco calibration preview" width="720">

This script has no CLI flags; adjust the constants near the top of the file.

```bash
python scripts/live_calibration/live_charuco_calibration.py
```


