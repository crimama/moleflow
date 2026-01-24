"""Generate zero-forgetting verification plots for the paper.

Outputs (PDF + PNG):
  - Paper_works/figures/zero_forgetting_bwt_verification.pdf
  - Paper_works/figures/zero_forgetting_bwt_verification.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    # Headless-safe backend (CI/servers).
    matplotlib.use("Agg")

    # Source: Paper_works/documents/Outline/4.Experiments_Tables.md (Table 5)
    task_ids = list(range(15))
    task_classes = [
        "bottle",
        "cable",
        "capsule",
        "carpet",
        "grid",
        "hazelnut",
        "leather",
        "metal_nut",
        "pill",
        "screw",
        "tile",
        "toothbrush",
        "transistor",
        "wood",
        "zipper",
    ]

    after_mean = np.array(
        [
            99.2,
            96.8,
            97.4,
            99.5,
            99.8,
            99.5,
            100.0,
            99.3,
            97.8,
            94.2,
            99.2,
            98.9,
            97.2,
            98.7,
            98.4,
        ],
        dtype=float,
    )
    after_std = np.array(
        [0.2, 0.4, 0.3, 0.1, 0.1, 0.1, 0.0, 0.2, 0.3, 0.6, 0.2, 0.2, 0.4, 0.3, 0.3],
        dtype=float,
    )
    final_mean = after_mean.copy()
    final_std = after_std.copy()

    # Source: Paper_works/documents/Outline/4.Experiments_Tables.md (Table 5, baseline comparison)
    methods = ["Fine-tune", "EWC", "ReplayCAD", "DeCoFlow"]
    final_iauc = np.array([63.39, 63.67, 94.0, 98.05], dtype=float)
    bwt_iauc = np.array([-17.30, -9.25, -0.4, 0.0], dtype=float)
    fm_iauc = np.array([21.33, 16.26, 1.5, 0.0], dtype=float)
    approx_mask = np.array(
        [False, False, True, False], dtype=bool
    )  # ReplayCAD values are approximate (~)

    out_dir = Path("Paper_works/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "zero_forgetting_bwt_verification.pdf"
    out_png = out_dir / "zero_forgetting_bwt_verification.png"

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
        }
    )

    fig = plt.figure(figsize=(11.5, 4.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])

    # (a) Task retention (paired) plot: After training vs final.
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(task_ids))

    # Use a paired (dumbbell) visualization to avoid overlapping curves
    # when retention is perfect (after == final).
    x_left = x - 0.12
    x_right = x + 0.12

    # Connect pairs (after -> final) per task.
    for i in range(len(x)):
        ax1.plot(
            [x_left[i], x_right[i]],
            [after_mean[i], final_mean[i]],
            color="0.65",
            lw=1.2,
            zorder=1,
        )

    ax1.errorbar(
        x_left,
        after_mean,
        yerr=after_std,
        fmt="o",
        ms=5,
        capsize=2,
        color="#1f77b4",
        label="After training",
        zorder=3,
    )
    ax1.errorbar(
        x_right,
        final_mean,
        yerr=final_std,
        fmt="s",
        ms=5,
        capsize=2,
        color="#2ca02c",
        label="Final (after all tasks)",
        zorder=3,
    )

    ax1.axhline(after_mean.mean(), color="0.35", lw=1.0, ls=":", zorder=0)

    ax1.set_title("(a) Task Retention (DeCoFlow, MVTec-AD 15 tasks)")
    ax1.set_xlabel("Task (class)")
    ax1.set_ylabel("Image AUC (%, mean ± std)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{i}\n{c}" for i, c in zip(task_ids, task_classes)])
    ax1.set_ylim(92.0, 100.6)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="lower left", frameon=True, ncols=2)

    # (b) Baseline comparison: BWT and FM.
    ax2 = fig.add_subplot(gs[0, 1])
    m = np.arange(len(methods))
    w = 0.36

    b1 = ax2.bar(
        m - w / 2,
        bwt_iauc,
        width=w,
        color="#1f77b4",
        label="BWT (I-AUC)",
        edgecolor="black",
        linewidth=0.6,
    )
    b2 = ax2.bar(
        m + w / 2,
        fm_iauc,
        width=w,
        color="#ff7f0e",
        label="FM (I-AUC)",
        edgecolor="black",
        linewidth=0.6,
    )

    # Hatch approximate values (ReplayCAD).
    for i, is_approx in enumerate(approx_mask):
        if not is_approx:
            continue
        b1[i].set_hatch("//")
        b2[i].set_hatch("//")

    ax2.axhline(0.0, color="black", lw=0.9)
    ax2.set_title("(b) Forgetting Metrics vs Baselines")
    ax2.set_ylabel("% (lower is better for FM; closer to 0 is better for BWT)")
    ax2.set_xticks(m)
    ax2.set_xticklabels(methods)
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right", frameon=True)

    # Annotate Final I-AUC above each method.
    for i, (val, is_approx) in enumerate(zip(final_iauc, approx_mask)):
        prefix = "~" if is_approx else ""
        ax2.text(
            i,
            max(fm_iauc[i], 0.0) + 1.0,
            f"Final I-AUC: {prefix}{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle(
        "Zero Forgetting Verification (MVTec-AD, 15-task continual learning)", y=1.02
    )
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
