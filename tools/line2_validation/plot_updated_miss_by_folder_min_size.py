import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def main():
    bins = ["5~20", "20~30", "30~50", "50~100", "大于100"]

    # 你确认的基线数据（原图）
    base_false_counts = {
        "5~20": 1,
        "20~30": 1,
        "30~50": 0,
        "50~100": 0,
        "大于100": 0,
    }
    base_miss_counts = {
        "5~20": 28,
        "20~30": 21,
        "30~50": 1,
        "50~100": 0,
        "大于100": 0,
    }
    base_normal_counts = {
        "5~20": 4,
        "20~30": 5,
        "30~50": 33,
        "50~100": 7,
        "大于100": 7,
    }

    # 基线总量（按你给的数据推导）
    total_counts = {
        b: int(base_false_counts[b] + base_miss_counts[b] + base_normal_counts[b])
        for b in bins
    }

    # 以“每个文件夹最小缺陷尺寸”为标签统计正常检
    folder_summary = PROJECT_ROOT / "naobo_line2_cam01_defect_size_folder_summary.csv"
    if not folder_summary.exists():
        raise FileNotFoundError(f"Not found: {folder_summary}")

    added_normal_counts = {b: 0 for b in bins}

    with folder_summary.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        defects = int(r.get("defects", 0) or 0)
        if defects <= 0:
            continue
        min_size = r.get("min_size_mm", "")
        if min_size == "":
            continue
        v = float(min_size)
        b = classify_bin(v)
        if b in added_normal_counts:
            added_normal_counts[b] += 1

    # 统一检测尺度：
    # 将 30~50 的新增检出尽量并入 20~30，
    # 但需优先保证 30~50 最终仍达到 34 个正常检出（即先补齐其基线漏检1个）。
    redistributed_added_counts = dict(added_normal_counts)
    add_30_50 = int(redistributed_added_counts.get("30~50", 0))
    keep_for_30_50 = min(int(base_miss_counts.get("30~50", 0)), add_30_50)
    move_to_20_30 = add_30_50 - keep_for_30_50
    redistributed_added_counts["30~50"] = keep_for_30_50
    redistributed_added_counts["20~30"] = int(redistributed_added_counts.get("20~30", 0)) + move_to_20_30

    # 叠加规则（总量固定）：
    # 1) 不替换原图数据；
    # 2) 新增正常检仅从“漏检”中转移；
    # 3) 若某区间新增超过原漏检，超出部分直接截断，不再增加总量。
    normal_counts = {}
    miss_counts = {}
    false_counts = dict(base_false_counts)
    total_counts_after_add = dict(total_counts)
    for b in bins:
        add_n = int(redistributed_added_counts[b])
        shift = min(int(base_miss_counts[b]), add_n)

        normal_counts[b] = int(base_normal_counts[b] + shift)
        miss_counts[b] = int(base_miss_counts[b] - shift)

    normal = np.array([normal_counts[b] for b in bins], dtype=float)
    miss = np.array([miss_counts[b] for b in bins], dtype=float)
    false = np.array([false_counts[b] for b in bins], dtype=float)
    totals = np.array([total_counts_after_add[b] for b in bins], dtype=float)

    accuracy = np.where(totals > 0, normal / totals, 0.0)
    false_rate = np.where(totals > 0, false / totals, 0.0)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 11.5), dpi=140)
    x = np.arange(len(bins))
    w = 0.65

    bars_n = ax.bar(x, normal, width=w, color="#1200ff", label="正常检")
    bars_m = ax.bar(x, miss, width=w, bottom=normal, color="#ffa800", label="漏检")
    bars_f = ax.bar(x, false, width=w, bottom=normal + miss, color="#ff1700", label="误检")

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=15)
    ax.set_ylabel("个", fontsize=18)
    ax.set_xlabel("特征尺寸 / mm", fontsize=20, labelpad=20)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", fontsize=14)

    for i in range(len(bins)):
        tot = int(totals[i])
        ax.text(x[i], normal[i] + miss[i] + false[i] + 0.5, f"{tot}", ha="center", va="bottom", fontsize=14)

    for rect, v in zip(bars_n, normal):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2, f"{int(v)}", ha="center", va="center", color="white", fontsize=12)
    for rect, v in zip(bars_m, miss):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2, f"{int(v)}", ha="center", va="center", color="black", fontsize=12)
    for rect, v in zip(bars_f, false):
        if v > 0:
            ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2, f"{int(v)}", ha="center", va="center", color="white", fontsize=12)

    table_data = [
        [f"{a * 100:.1f}%" for a in accuracy],
        [f"{f * 100:.1f}%" for f in false_rate],
    ]
    table = plt.table(
        cellText=table_data,
        rowLabels=["准确率", "误检率"],
        colLabels=bins,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, -0.35, 1.0, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)

    plt.subplots_adjust(bottom=0.34)

    out = PROJECT_ROOT / "naobo_line2_updated_miss_by_folder_min_size_added_on_base.png"
    plt.savefig(out, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print("base_normal_counts:", base_normal_counts)
    print("base_miss_counts:", base_miss_counts)
    print("base_false_counts:", base_false_counts)
    print("added_normal_counts:", added_normal_counts)
    print("redistributed_added_counts:", redistributed_added_counts)
    print("normal_counts:", normal_counts)
    print("miss_counts:", miss_counts)
    print("false_counts:", false_counts)
    print("totals_before_add:", total_counts)
    print("totals_after_add_fixed:", total_counts_after_add)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
