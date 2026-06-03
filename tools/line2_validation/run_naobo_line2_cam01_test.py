import json
import os
import argparse
import sys
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fused_image_processor
import processing_worker


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_rois(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        return []
    best_key = max(data.keys(), key=lambda k: data[k].get("source_image_count", 0))
    return data[best_key].get("averaged_rois", [])


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Batch test naobo_line2 cam01 images")
    parser.add_argument(
        "--insect-filter",
        choices=["on", "off"],
        default="off",
        help="Enable or disable flying insect filter from processing_worker",
    )
    args = parser.parse_args()

    root = PROJECT_ROOT / "naobo_line2"
    config_path = PROJECT_ROOT / "config2.yaml"
    roi_path = Path(__file__).with_name("cam01_2026-04-10T10-23-56-999_0001_roi_averaged_by_group.json")

    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Not found: {config_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"Not found: {roi_path}")

    config = load_config(config_path)
    pixels_per_mm = float((config.get("system_params", {}) or {}).get("pixels_per_mm", 2.4))
    rois = load_rois(roi_path)
    if not rois:
        print("[WARN] ROI empty, results may all be OK")

    # 固定使用明场参数: config2.yaml -> hough_inspector_params
    mode = 1

    rows = []
    total_images = 0
    total_defects = 0
    total_ng_images = 0

    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])
    for sub in subfolders:
        cam01 = sub / "cam01"
        if not cam01.exists() or not cam01.is_dir():
            rows.append(
                {
                    "folder": sub.name,
                    "images": 0,
                    "ng_images": 0,
                    "defects": 0,
                    "skipped": 0,
                    "note": "cam01 missing",
                }
            )
            continue

        image_count = 0
        ng_count = 0
        defect_count = 0
        skipped = 0

        for img_path in iter_images(cam01):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                skipped += 1
                continue
            image_count += 1
            report, _ = fused_image_processor.process_image(gray, rois, config, mode=mode)
            if args.insect_filter == "on":
                processing_worker.filter_flying_insects(report, gray, pixels_per_mm)
            defects = len((report or {}).get("defects", []))
            defect_count += defects
            if defects > 0 or (report or {}).get("image_status") == "NG":
                ng_count += 1

        rows.append(
            {
                "folder": sub.name,
                "images": image_count,
                "ng_images": ng_count,
                "defects": defect_count,
                "skipped": skipped,
                "note": "",
            }
        )
        total_images += image_count
        total_defects += defect_count
        total_ng_images += ng_count

    # 输出汇总到文件
    suffix = "with_insect_filter" if args.insect_filter == "on" else "no_insect_filter"
    out_json = PROJECT_ROOT / f"naobo_line2_cam01_defect_report_{suffix}.json"
    out_csv = PROJECT_ROOT / f"naobo_line2_cam01_defect_report_{suffix}.csv"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "config": str(config_path),
                "roi": str(roi_path),
                "insect_filter": args.insect_filter,
                "mode": mode,
                "totals": {
                    "folders": len(rows),
                    "images": total_images,
                    "ng_images": total_ng_images,
                    "defects": total_defects,
                },
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("folder,images,ng_images,defects,skipped,note\n")
        for r in rows:
            note = (r["note"] or "").replace(",", " ")
            f.write(
                f"{r['folder']},{r['images']},{r['ng_images']},{r['defects']},{r['skipped']},{note}\n"
            )

    # 控制台打印
    print("folder | images | ng_images | defects | skipped")
    for r in rows:
        print(
            f"{r['folder']} | {r['images']} | {r['ng_images']} | {r['defects']} | {r['skipped']}"
        )
    print("-" * 72)
    print(
        f"TOTAL | {total_images} | {total_ng_images} | {total_defects} | {sum(r['skipped'] for r in rows)}"
    )
    print(f"saved: {out_json}")
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
