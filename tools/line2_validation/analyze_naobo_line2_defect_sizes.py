import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fused_image_processor


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


def safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def classify_bin(size_mm: float) -> str:
    if size_mm < 5:
        return "小于5"
    if size_mm < 20:
        return "5~20"
    if size_mm < 30:
        return "20~30"
    if size_mm < 50:
        return "30~50"
    if size_mm < 100:
        return "50~100"
    return "大于100"


def get_char_size_mm(defect: dict):
    loc = defect.get("location", {}) or {}
    w_mm = safe_float(loc.get("width_mm"))
    l_mm = safe_float(loc.get("length_mm"))

    cands = []
    if w_mm is not None:
        cands.append(w_mm)
    if l_mm is not None:
        cands.append(l_mm)

    if cands:
        return max(cands), w_mm, l_mm

    alt = safe_float(defect.get("size_mm"))
    if alt is not None:
        return alt, None, None

    return None, w_mm, l_mm


def save_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def plot_size_bin_stacked_by_type(bin_type_counts: dict, out_png: Path):
    # 尽量兼容 Windows 中文显示
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    bins_order = ["小于5", "5~20", "20~30", "30~50", "50~100", "大于100"]
    defect_types = sorted({t for m in bin_type_counts.values() for t in m.keys()})
    if not defect_types:
        defect_types = ["UNKNOWN"]

    x = np.arange(len(bins_order))
    bottom = np.zeros(len(bins_order), dtype=float)

    colors = {
        "Q": "#0057ff",
        "B": "#ff9800",
        "L": "#f44336",
        "E": "#4caf50",
        "UNKNOWN": "#9e9e9e",
    }

    plt.figure(figsize=(12, 8), dpi=120)
    for t in defect_types:
        vals = np.array([bin_type_counts.get(b, {}).get(t, 0) for b in bins_order], dtype=float)
        bars = plt.bar(x, vals, bottom=bottom, label=t, color=colors.get(t, None), width=0.65)
        for i, v in enumerate(vals):
            if v > 0:
                plt.text(i, bottom[i] + v / 2, f"{int(v)}", ha="center", va="center", fontsize=10, color="white")
        bottom += vals

    for i, tot in enumerate(bottom):
        if tot > 0:
            plt.text(i, tot + 0.3, f"{int(tot)}", ha="center", va="bottom", fontsize=12, color="black")

    plt.xticks(x, bins_order, fontsize=12)
    plt.ylabel("缺陷数量", fontsize=13)
    plt.xlabel("特征尺寸 / mm", fontsize=16)
    plt.title("naobo_line2 全量缺陷尺寸分布（按缺陷类型堆叠）", fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(title="缺陷类型")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_folder_bin_stacked(folder_bin_counts: dict, out_png: Path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    bins_order = ["小于5", "5~20", "20~30", "30~50", "50~100", "大于100"]
    folders = sorted(folder_bin_counts.keys())

    x = np.arange(len(folders))
    bottom = np.zeros(len(folders), dtype=float)
    colors = {
        "小于5": "#8bc34a",
        "5~20": "#03a9f4",
        "20~30": "#ff9800",
        "30~50": "#3f51b5",
        "50~100": "#9c27b0",
        "大于100": "#f44336",
    }

    plt.figure(figsize=(18, 8), dpi=120)
    for b in bins_order:
        vals = np.array([folder_bin_counts.get(fd, {}).get(b, 0) for fd in folders], dtype=float)
        plt.bar(x, vals, bottom=bottom, label=b, color=colors.get(b, None), width=0.75)
        bottom += vals

    plt.xticks(x, folders, rotation=45, ha="right", fontsize=9)
    plt.ylabel("缺陷数量", fontsize=13)
    plt.xlabel("子文件夹", fontsize=13)
    plt.title("naobo_line2 每子文件夹缺陷尺寸分布（按尺寸区间堆叠）", fontsize=15)
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend(title="尺寸区间", ncol=3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze defect sizes (mm) in naobo_line2 cam01")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "naobo_line2"), help="Root folder containing subfolders")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config2.yaml"), help="YAML config path")
    parser.add_argument(
        "--roi",
        default=str(Path(__file__).with_name("cam01_2026-04-10T10-23-56-999_0001_roi_averaged_by_group.json")),
        help="ROI json path",
    )
    parser.add_argument(
        "--output-prefix",
        default=str(PROJECT_ROOT / "naobo_line2_cam01_defect_size"),
        help="Output filename prefix",
    )
    args = parser.parse_args()

    root = Path(args.root)
    config_path = Path(args.config)
    roi_path = Path(args.roi)

    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")
    if not config_path.exists():
        raise FileNotFoundError(f"Not found: {config_path}")
    if not roi_path.exists():
        raise FileNotFoundError(f"Not found: {roi_path}")

    config = load_config(config_path)
    rois = load_rois(roi_path)
    if not rois:
        print("[WARN] ROI empty, results may all be OK")

    all_defect_rows = []
    folder_summary = []
    folder_bin_counts = {}
    bin_type_counts = {}

    subfolders = sorted([p for p in root.iterdir() if p.is_dir()])

    total_images = 0
    total_defects = 0
    skipped_images = 0

    for sub in subfolders:
        cam01 = sub / "cam01"
        if not cam01.exists() or not cam01.is_dir():
            folder_summary.append(
                {
                    "folder": sub.name,
                    "images": 0,
                    "defects": 0,
                    "mean_size_mm": "",
                    "median_size_mm": "",
                    "min_size_mm": "",
                    "max_size_mm": "",
                    "note": "cam01 missing",
                }
            )
            folder_bin_counts[sub.name] = {}
            continue

        defect_sizes = []
        defect_count = 0
        image_count = 0
        folder_bin_counts[sub.name] = {}

        for img_path in iter_images(cam01):
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                skipped_images += 1
                continue

            image_count += 1
            report, _ = fused_image_processor.process_image(gray, rois, config, mode=1)
            defects = (report or {}).get("defects", [])

            for d in defects:
                char_size_mm, w_mm, l_mm = get_char_size_mm(d)
                if char_size_mm is None:
                    continue

                d_type = str(d.get("type", "UNKNOWN") or "UNKNOWN").upper()
                b = classify_bin(char_size_mm)

                defect_count += 1
                defect_sizes.append(char_size_mm)
                folder_bin_counts[sub.name][b] = int(folder_bin_counts[sub.name].get(b, 0)) + 1
                if b not in bin_type_counts:
                    bin_type_counts[b] = {}
                bin_type_counts[b][d_type] = int(bin_type_counts[b].get(d_type, 0)) + 1

                loc = d.get("location", {}) or {}
                all_defect_rows.append(
                    {
                        "folder": sub.name,
                        "image": img_path.name,
                        "type": d_type,
                        "roi_idx": d.get("roi_idx", ""),
                        "x_px": loc.get("x", ""),
                        "y_px": loc.get("y", ""),
                        "width_mm": round(w_mm, 4) if w_mm is not None else "",
                        "length_mm": round(l_mm, 4) if l_mm is not None else "",
                        "char_size_mm": round(char_size_mm, 4),
                        "size_bin": b,
                    }
                )

        total_images += image_count
        total_defects += defect_count

        if defect_sizes:
            arr = np.array(defect_sizes, dtype=float)
            folder_summary.append(
                {
                    "folder": sub.name,
                    "images": image_count,
                    "defects": defect_count,
                    "mean_size_mm": round(float(np.mean(arr)), 4),
                    "median_size_mm": round(float(np.median(arr)), 4),
                    "min_size_mm": round(float(np.min(arr)), 4),
                    "max_size_mm": round(float(np.max(arr)), 4),
                    "note": "",
                }
            )
        else:
            folder_summary.append(
                {
                    "folder": sub.name,
                    "images": image_count,
                    "defects": 0,
                    "mean_size_mm": "",
                    "median_size_mm": "",
                    "min_size_mm": "",
                    "max_size_mm": "",
                    "note": "no defects",
                }
            )

    prefix = args.output_prefix
    all_csv = Path(f"{prefix}_all_defects.csv")
    folder_csv = Path(f"{prefix}_folder_summary.csv")
    stats_json = Path(f"{prefix}_stats.json")
    png_overall = Path(f"{prefix}_overall_stacked_by_type.png")
    png_folder = Path(f"{prefix}_folder_stacked_by_bin.png")

    save_csv(
        all_csv,
        all_defect_rows,
        headers=[
            "folder",
            "image",
            "type",
            "roi_idx",
            "x_px",
            "y_px",
            "width_mm",
            "length_mm",
            "char_size_mm",
            "size_bin",
        ],
    )
    save_csv(
        folder_csv,
        folder_summary,
        headers=[
            "folder",
            "images",
            "defects",
            "mean_size_mm",
            "median_size_mm",
            "min_size_mm",
            "max_size_mm",
            "note",
        ],
    )

    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "config": str(config_path),
                "roi": str(roi_path),
                "totals": {
                    "folders": len(subfolders),
                    "images": total_images,
                    "defects": total_defects,
                    "skipped_images": skipped_images,
                },
                "bin_type_counts": bin_type_counts,
                "folder_bin_counts": folder_bin_counts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    plot_size_bin_stacked_by_type(bin_type_counts, png_overall)
    plot_folder_bin_stacked(folder_bin_counts, png_folder)

    print(f"total_folders={len(subfolders)}")
    print(f"total_images={total_images}")
    print(f"total_defects={total_defects}")
    print(f"skipped_images={skipped_images}")
    print(f"saved: {all_csv}")
    print(f"saved: {folder_csv}")
    print(f"saved: {stats_json}")
    print(f"saved: {png_overall}")
    print(f"saved: {png_folder}")


if __name__ == "__main__":
    main()
