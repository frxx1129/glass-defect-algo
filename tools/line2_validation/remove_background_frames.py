import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_rois(roi_path: Path):
    with roi_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        return []
    best_key = max(data.keys(), key=lambda k: data[k].get("source_image_count", 0))
    rois = data[best_key].get("averaged_rois", [])
    out = []
    for r in rois:
        out.append(
            {
                "x": int(r.get("x", 0)),
                "y": int(r.get("y", 0)),
                "width": int(r.get("width", 0)),
                "height": int(r.get("height", 0)),
            }
        )
    return out


def iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def entropy_u8(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist_sum = float(hist.sum())
    if hist_sum <= 0:
        return 0.0
    p = hist / hist_sum
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log2(p)))


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    return x, y, w, h


@dataclass
class FrameMetrics:
    path: Path
    std_mean: float
    lap_var_mean: float
    edge_ratio_mean: float
    entropy_mean: float
    info_score: float = 0.0
    is_background: bool = False
    reason: str = ""


def calc_metrics(gray: np.ndarray, rois):
    H, W = gray.shape[:2]
    std_vals = []
    lap_vals = []
    edge_vals = []
    ent_vals = []

    if not rois:
        rois = [{"x": 0, "y": 0, "width": W, "height": H}]

    for r in rois:
        x, y, w, h = clamp_roi(r["x"], r["y"], r["width"], r["height"], W, H)
        if w < 8 or h < 8:
            continue
        crop = gray[y : y + h, x : x + w]
        std_vals.append(float(np.std(crop)))
        lap_var = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        lap_vals.append(lap_var)

        med = float(np.median(crop))
        low = int(max(0, med * 0.66))
        high = int(min(255, med * 1.33 + 10))
        edges = cv2.Canny(crop, low, high)
        edge_ratio = float(np.count_nonzero(edges)) / float(edges.size)
        edge_vals.append(edge_ratio)

        ent_vals.append(entropy_u8(crop))

    if not std_vals:
        return 0.0, 0.0, 0.0, 0.0

    return (
        float(np.mean(std_vals)),
        float(np.mean(lap_vals)),
        float(np.mean(edge_vals)),
        float(np.mean(ent_vals)),
    )


def q_norm(values, q10, q90):
    den = max(1e-9, (q90 - q10))
    arr = (values - q10) / den
    return np.clip(arr, 0.0, 1.0)


def detect_background_candidates(metrics_list):
    stds = np.array([m.std_mean for m in metrics_list], dtype=float)
    laps = np.array([m.lap_var_mean for m in metrics_list], dtype=float)
    edges = np.array([m.edge_ratio_mean for m in metrics_list], dtype=float)
    ents = np.array([m.entropy_mean for m in metrics_list], dtype=float)

    s10, s90 = np.percentile(stds, [10, 90])
    l10, l90 = np.percentile(laps, [10, 90])
    e10, e90 = np.percentile(edges, [10, 90])
    h10, h90 = np.percentile(ents, [10, 90])

    n_std = q_norm(stds, s10, s90)
    n_lap = q_norm(laps, l10, l90)
    n_edge = q_norm(edges, e10, e90)
    n_ent = q_norm(ents, h10, h90)

    scores = 0.35 * n_std + 0.35 * n_lap + 0.2 * n_edge + 0.1 * n_ent

    p20_score = float(np.percentile(scores, 20))
    score_thr = min(0.28, max(0.08, p20_score))

    s30 = float(np.percentile(stds, 30))
    l30 = float(np.percentile(laps, 30))
    e30 = float(np.percentile(edges, 30))

    for i, m in enumerate(metrics_list):
        m.info_score = float(scores[i])

        votes = 0
        if m.std_mean <= s30:
            votes += 1
        if m.lap_var_mean <= l30:
            votes += 1
        if m.edge_ratio_mean <= e30:
            votes += 1

        if m.info_score <= score_thr and votes >= 2:
            m.is_background = True
            m.reason = f"score<=thr({m.info_score:.4f}<={score_thr:.4f}), votes={votes}"
        else:
            m.is_background = False
            m.reason = f"score={m.info_score:.4f}, votes={votes}"

    return {
        "score_threshold": score_thr,
        "std_p30": s30,
        "lap_p30": l30,
        "edge_p30": e30,
    }


def process_cam_folder(cam_folder: Path, rois, apply: bool, mode: str, quarantine_root: Path):
    metrics = []
    for img in iter_images(cam_folder):
        gray = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        s, l, e, h = calc_metrics(gray, rois)
        metrics.append(
            FrameMetrics(
                path=img,
                std_mean=s,
                lap_var_mean=l,
                edge_ratio_mean=e,
                entropy_mean=h,
            )
        )

    if not metrics:
        return {
            "folder": str(cam_folder),
            "total": 0,
            "background": 0,
            "removed": 0,
            "thresholds": {},
            "rows": [],
        }

    thresholds = detect_background_candidates(metrics)

    removed = 0
    rows = []
    for m in metrics:
        action = "keep"
        if m.is_background:
            if apply:
                if mode == "delete":
                    m.path.unlink(missing_ok=True)
                    action = "deleted"
                else:
                    rel_name = m.path.name
                    dst_dir = quarantine_root / cam_folder.parent.name / cam_folder.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(m.path), str(dst_dir / rel_name))
                    action = "moved"
                removed += 1
            else:
                action = "candidate"

        rows.append(
            {
                "folder": str(cam_folder),
                "file": m.path.name,
                "std_mean": round(m.std_mean, 6),
                "lap_var_mean": round(m.lap_var_mean, 6),
                "edge_ratio_mean": round(m.edge_ratio_mean, 8),
                "entropy_mean": round(m.entropy_mean, 6),
                "info_score": round(m.info_score, 6),
                "is_background": int(m.is_background),
                "action": action,
                "reason": m.reason,
            }
        )

    return {
        "folder": str(cam_folder),
        "total": len(metrics),
        "background": sum(1 for x in metrics if x.is_background),
        "removed": removed,
        "thresholds": thresholds,
        "rows": rows,
    }


def discover_cam_folders(root: Path, cam_name: str):
    cam_folders = []
    if root.name.lower() == cam_name.lower() and root.is_dir():
        cam_folders.append(root)
    else:
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            c = sub / cam_name
            if c.exists() and c.is_dir():
                cam_folders.append(c)
    return cam_folders


def main():
    parser = argparse.ArgumentParser(description="Remove pure-background frames in replay folders")
    parser.add_argument("--root", default=str(PROJECT_ROOT / "naobo_line2"), help="Root folder (e.g. naobo_line2)")
    parser.add_argument("--cam", default="cam01", help="Camera folder name")
    parser.add_argument(
        "--roi",
        default=str(Path(__file__).with_name("cam01_2026-04-10T10-23-56-999_0001_roi_averaged_by_group.json")),
        help="ROI json path",
    )
    parser.add_argument("--apply", action="store_true", help="Actually apply removal")
    parser.add_argument(
        "--mode",
        choices=["move", "delete"],
        default="move",
        help="When --apply: move to quarantine or delete directly",
    )
    parser.add_argument(
        "--quarantine-dir",
        default=str(PROJECT_ROOT / "removed_background_candidates"),
        help="Quarantine directory for move mode",
    )
    parser.add_argument(
        "--report-prefix",
        default=str(PROJECT_ROOT / "background_cleanup_report"),
        help="Output report prefix",
    )
    args = parser.parse_args()

    root = Path(args.root)
    roi_path = Path(args.roi)
    quarantine_root = Path(args.quarantine_dir)

    if not root.exists():
        raise FileNotFoundError(f"Not found: {root}")
    if not roi_path.exists():
        raise FileNotFoundError(f"Not found: {roi_path}")

    rois = load_rois(roi_path)
    cam_folders = discover_cam_folders(root, args.cam)
    if not cam_folders:
        print(f"No {args.cam} folders found under {root}")
        return

    all_rows = []
    folder_summary = []

    total_images = 0
    total_bg = 0
    total_removed = 0

    for cam_folder in cam_folders:
        res = process_cam_folder(cam_folder, rois, args.apply, args.mode, quarantine_root)

        folder_summary.append(
            {
                "folder": res["folder"],
                "total_images": res["total"],
                "background_candidates": res["background"],
                "removed": res["removed"],
                "score_threshold": round(float(res["thresholds"].get("score_threshold", 0.0)), 6)
                if res["thresholds"]
                else "",
                "std_p30": round(float(res["thresholds"].get("std_p30", 0.0)), 6)
                if res["thresholds"]
                else "",
                "lap_p30": round(float(res["thresholds"].get("lap_p30", 0.0)), 6)
                if res["thresholds"]
                else "",
                "edge_p30": round(float(res["thresholds"].get("edge_p30", 0.0)), 8)
                if res["thresholds"]
                else "",
            }
        )
        all_rows.extend(res["rows"])

        total_images += res["total"]
        total_bg += res["background"]
        total_removed += res["removed"]

        print(
            f"{cam_folder}: total={res['total']} bg_candidates={res['background']} removed={res['removed']}"
        )

    prefix = args.report_prefix
    details_csv = Path(f"{prefix}_details.csv")
    summary_csv = Path(f"{prefix}_summary.csv")
    summary_json = Path(f"{prefix}_summary.json")

    with details_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "file",
                "std_mean",
                "lap_var_mean",
                "edge_ratio_mean",
                "entropy_mean",
                "info_score",
                "is_background",
                "action",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "folder",
                "total_images",
                "background_candidates",
                "removed",
                "score_threshold",
                "std_p30",
                "lap_p30",
                "edge_p30",
            ],
        )
        writer.writeheader()
        writer.writerows(folder_summary)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root": str(root),
                "cam": args.cam,
                "roi": str(roi_path),
                "apply": bool(args.apply),
                "mode": args.mode,
                "quarantine_dir": str(quarantine_root),
                "totals": {
                    "folders": len(folder_summary),
                    "images": total_images,
                    "background_candidates": total_bg,
                    "removed": total_removed,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("-" * 72)
    print(f"folders={len(folder_summary)} images={total_images} candidates={total_bg} removed={total_removed}")
    print(f"saved: {details_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    if args.apply and args.mode == "move":
        print(f"quarantine: {quarantine_root}")


if __name__ == "__main__":
    main()
