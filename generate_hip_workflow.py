#!/usr/bin/env python3
"""
Generate a technical workflow diagram for CUDA/Triton -> HIP optimization cycle.

Output:
    hip_migration_workflow.png (default)

Usage:
    python generate_hip_workflow.py
    python generate_hip_workflow.py --output my_workflow.png --dpi 200
"""

import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


def add_box(ax, xy, w, h, text, fc="#EAF2FF", ec="#2B5AA6", fontsize=10):
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.6,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans",
    )
    return box


def add_diamond(ax, center, w, h, text, fc="#FFF4D9", ec="#A66A00", fontsize=10):
    cx, cy = center
    pts = [
        (cx, cy + h / 2),
        (cx + w / 2, cy),
        (cx, cy - h / 2),
        (cx - w / 2, cy),
    ]
    diamond = Polygon(pts, closed=True, linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(diamond)
    ax.text(
        cx, cy, text, ha="center", va="center", fontsize=fontsize, family="DejaVu Sans"
    )
    return diamond


def arrow(
    ax, p1, p2, text=None, text_offset=(0, 0), color="#333333", text_fontsize=9
):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arr)
    if text:
        mx = (p1[0] + p2[0]) / 2 + text_offset[0]
        my = (p1[1] + p2[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, fontsize=text_fontsize, color=color, family="DejaVu Sans")
    return arr


def curved_arrow(
    ax, p1, p2, rad=0.2, text=None, text_xy=None, color="#333333", text_fontsize=9
):
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=1.5,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)
    if text and text_xy:
        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            fontsize=text_fontsize,
            color=color,
            family="DejaVu Sans",
        )
    return arr


def main(output, dpi):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.5,
        0.95,
        "Technical Workflow: CUDA/Triton -> HIP Optimization Loop",
        ha="center",
        va="center",
        fontsize=16,
        weight="bold",
        family="DejaVu Sans",
    )

    # Nodes
    add_box(
        ax,
        (0.05, 0.72),
        0.24,
        0.12,
        "0) Baseline on NVIDIA\n(CUDA/Triton)\nRecord latency/throughput",
        fc="#F0F0F0",
        ec="#555555",
    )

    add_box(
        ax,
        (0.36, 0.72),
        0.28,
        0.12,
        "1) Convert to HIP + HIP-compatible Triton\n(hipify/manual port + build/runtime updates)",
    )

    add_box(
        ax,
        (0.70, 0.72),
        0.24,
        0.12,
        "2) Run on AMD\nCollect end-to-end timing\n(+ correctness checks)",
        fc="#E9FCEB",
        ec="#2E7D32",
    )

    add_diamond(
        ax,
        center=(0.82, 0.54),
        w=0.30,
        h=0.18,
        text="AMD time >\nNVIDIA baseline\n(or target KPI)?$^{*}$",
    )

    add_box(
        ax,
        (0.40, 0.28),
        0.31,
        0.13,
        "3) Profile bottleneck(s)\n(kernel-level + memory/launch overhead)\nRewrite/tune slow HIP/Triton kernel(s)",
        fc="#FFEAEA",
        ec="#A62B2B",
    )

    add_box(
        ax,
        (0.06, 0.28),
        0.30,
        0.13,
        "4) Integrate optimized kernels\nRebuild + regression tests",
        fc="#FFF7E8",
        ec="#A66A00",
    )

    add_box(
        ax,
        (0.76, 0.28),
        0.22,
        0.13,
        "Exit: Performance target met\nShip / handover",
        fc="#E8F6FF",
        ec="#1F6FA8",
    )

    # Arrows (main path)
    arrow(ax, (0.29, 0.78), (0.36, 0.78))
    arrow(ax, (0.64, 0.78), (0.70, 0.78))
    arrow(ax, (0.82, 0.72), (0.82, 0.63))
    arrow(ax, (0.87, 0.54), (0.87, 0.41), text="No", text_offset=(0.015, 0.0))

    # Yes branch to optimize
    arrow(ax, (0.74, 0.54), (0.61, 0.41), text="Yes", text_offset=(-0.01, 0.02))
    arrow(ax, (0.40, 0.345), (0.36, 0.345))  # optimize -> integrate

    # Loop back to run step (straight connector)
    arrow(
        ax,
        (0.06, 0.345),  # from left side of integrate
        (0.70, 0.78),  # back to run step
        text="Iterative optimization loop",
        text_offset=(0.0, -0.06),
        text_fontsize=12,
    )

    # Footnote referenced by the decision block
    ax.text(
        0.03,
        0.08,
        r"$^{*}$ Loop condition: continue while AMD runtime is above CUDA baseline / agreed KPI.",
        ha="left",
        va="center",
        fontsize=10,
        family="DejaVu Sans",
        color="#222222",
    )

    plt.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved workflow diagram to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="hip_migration_workflow.png", help="Output image file path"
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output DPI")
    args = parser.parse_args()
    main(args.output, args.dpi)
