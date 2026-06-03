import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    stats_path = PROJECT_ROOT / "naobo_line2_cam01_defect_size_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    bins = ["5~20", "20~30", "30~50", "50~100", "大于100"]

    # 来自你的要求：各尺寸区间总量
    total_counts = {
        "5~20": 5,
        "20~30": 9,
        "30~50": 23,
        "50~100": 9,
        "大于100": 2,
    }

    # 统计检测到的数量（不区分缺陷类型）
    detected_counts = {b: 0 for b in bins}
    bin_type_counts = stats.get("bin_type_counts", {})
    for b in bins:
        type_map = bin_type_counts.get(b, {}) or {}
        detected_counts[b] = int(sum(int(v) for v in type_map.values()))

    detected = np.array([detected_counts[b] for b in bins], dtype=float)
    totals = np.array([total_counts[b] for b in bins], dtype=float)
    rates = np.where(totals > 0, detected / totals, 0.0)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 进一步延长图像高度，减少上下区域拥挤
    fig, ax = plt.subplots(figsize=(14, 13.4), dpi=140)
    x = np.arange(len(bins))
    bars = ax.bar(x, detected, color="#1565c0", width=0.62, label="检测数")

    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=13)
    ax.set_xlabel("特征尺寸 / mm", fontsize=15, labelpad=6)
    ax.set_ylabel("检测数量", fontsize=14)
    ax.set_title("全量缺陷尺寸分布（不区分缺陷类型）", fontsize=17)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_ylim(0, max(detected) + 4.0)

    # 在柱子上显示检测数与检出率
    for i, bar in enumerate(bars):
        h = bar.get_height()
        cx = bar.get_x() + bar.get_width() / 2
        ax.text(
            cx,
            h + 0.25,
            f"{int(h)}",
            ha="center",
            va="bottom",
            fontsize=12,
            color="black",
        )
        label = f"检出率 {rates[i] * 100:.1f}%\n({int(detected[i])}/{int(totals[i])})"
        # 统一放在柱内；小柱子使用更小字体并加深色底，确保可读
        y_in = max(h * 0.5, 0.35)
        fs = 8.6 if h < 4 else 10
        ax.text(
            cx,
            y_in,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            color="white",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#0d47a1", edgecolor="none", alpha=0.78),
        )

    # 底部表格：总量和检出率
    table_data = [
        [str(int(total_counts[b])) for b in bins],
        [f"{rates[i] * 100:.1f}%" for i in range(len(bins))],
    ]
    table = plt.table(
        cellText=table_data,
        rowLabels=["总量(帧)", "检出率"],
        colLabels=bins,
        cellLoc="center",
        rowLoc="center",
        bbox=[0.0, -0.37, 1.0, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # 显式增加底边距，防止表格被裁切
    plt.subplots_adjust(bottom=0.47)

    out_png = PROJECT_ROOT / "naobo_line2_cam01_defect_size_overall_with_detection_rate.png"
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print("bins:", bins)
    print("detected:", detected_counts)
    print("totals:", total_counts)
    print("rates:", {bins[i]: round(float(rates[i]) * 100.0, 2) for i in range(len(bins))})
    print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
