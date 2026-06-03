# Line2 offline validation tools

These scripts are optional production-support utilities for replaying `naobo_line2` cam01 images and summarizing detection results.

Defaults assume:

- replay data lives in `<project-root>/naobo_line2/`
- Line2 config is `<project-root>/config2.yaml`
- the included cam01 ROI sample is next to these scripts

Generated reports are written back to the project root and are ignored by Git.

Examples:

```powershell
python tools/line2_validation/run_naobo_line2_cam01_test.py --insect-filter off
python tools/line2_validation/remove_background_frames.py
python tools/line2_validation/analyze_naobo_line2_defect_sizes.py
```
