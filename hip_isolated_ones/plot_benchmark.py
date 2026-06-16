#!/usr/bin/env python3
"""
Bar chart for HIP isolated-ones benchmark (same layout as Colab notebook).

Usage:
    python plot_benchmark.py
    python plot_benchmark.py --input benchmark.txt --output hip_benchmark.png
    python plot_benchmark.py --output results.png --dpi 150
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

SIZES = [1024, 2048]
DENSITY_SCENARIOS = [
    (0.01, "sparse"),
    (0.10, "medium"),
    (0.90, "dense"),
]

# Display order (5e / 5g excluded — slower u32/shfl variants)
METHODS = [
    "1_interview_if_atomic",
    "1b_divergent_early_return",
    "2_uniform_block_reduce",
    "5_tile30_halo",
    "5a_tile30_nopad",
    "5b_tile16_halo",
    "5f_tile16_shfl",
    "5c_tile30_nbrcache",
    "5d_tile30_block30",
]

EXCLUDE_METHODS = {
    "5e_tile30_u32pack",
    "5g_tile30_u32pack_shfl",
    "0_cpu_reference",
}

# Default: RX 6800 XT run (2025-05-27), median ms
DEFAULT_ROWS = [
    # N=1024 sparse
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "1_interview_if_atomic", "median_ms": 0.17},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "1b_divergent_early_return", "median_ms": 0.17},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "2_uniform_block_reduce", "median_ms": 0.17},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5_tile30_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5a_tile30_nopad", "median_ms": 0.07},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5b_tile16_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5f_tile16_shfl", "median_ms": 0.07},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5c_tile30_nbrcache", "median_ms": 0.07},
    {"N": 1024, "density": 0.01, "regime": "sparse", "method": "5d_tile30_block30", "median_ms": 0.08},
    # N=1024 medium
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "1_interview_if_atomic", "median_ms": 0.20},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "1b_divergent_early_return", "median_ms": 0.21},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "2_uniform_block_reduce", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5_tile30_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5a_tile30_nopad", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5b_tile16_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5f_tile16_shfl", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5c_tile30_nbrcache", "median_ms": 0.07},
    {"N": 1024, "density": 0.10, "regime": "medium", "method": "5d_tile30_block30", "median_ms": 0.09},
    # N=1024 dense
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "1_interview_if_atomic", "median_ms": 0.16},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "1b_divergent_early_return", "median_ms": 0.16},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "2_uniform_block_reduce", "median_ms": 0.16},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5_tile30_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5a_tile30_nopad", "median_ms": 0.07},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5b_tile16_halo", "median_ms": 0.07},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5f_tile16_shfl", "median_ms": 0.07},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5c_tile30_nbrcache", "median_ms": 0.07},
    {"N": 1024, "density": 0.90, "regime": "dense", "method": "5d_tile30_block30", "median_ms": 0.09},
    # N=2048 sparse
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "1_interview_if_atomic", "median_ms": 0.31},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "1b_divergent_early_return", "median_ms": 0.20},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "2_uniform_block_reduce", "median_ms": 0.12},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5_tile30_halo", "median_ms": 0.10},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5a_tile30_nopad", "median_ms": 0.10},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5b_tile16_halo", "median_ms": 0.14},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5f_tile16_shfl", "median_ms": 0.14},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5c_tile30_nbrcache", "median_ms": 0.11},
    {"N": 2048, "density": 0.01, "regime": "sparse", "method": "5d_tile30_block30", "median_ms": 0.14},
    # N=2048 medium
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "1_interview_if_atomic", "median_ms": 0.45},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "1b_divergent_early_return", "median_ms": 0.16},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "2_uniform_block_reduce", "median_ms": 0.12},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5_tile30_halo", "median_ms": 0.10},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5a_tile30_nopad", "median_ms": 0.10},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5b_tile16_halo", "median_ms": 0.14},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5f_tile16_shfl", "median_ms": 0.14},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5c_tile30_nbrcache", "median_ms": 0.11},
    {"N": 2048, "density": 0.10, "regime": "medium", "method": "5d_tile30_block30", "median_ms": 0.15},
    # N=2048 dense
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "1_interview_if_atomic", "median_ms": 0.29},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "1b_divergent_early_return", "median_ms": 0.29},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "2_uniform_block_reduce", "median_ms": 0.12},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5_tile30_halo", "median_ms": 0.10},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5a_tile30_nopad", "median_ms": 0.10},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5b_tile16_halo", "median_ms": 0.14},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5f_tile16_shfl", "median_ms": 0.14},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5c_tile30_nbrcache", "median_ms": 0.12},
    {"N": 2048, "density": 0.90, "regime": "dense", "method": "5d_tile30_block30", "median_ms": 0.15},
]

SECTION_RE = re.compile(
    r"^=== N=(\d+) density=([\d.]+) \((\w+)\) ===$"
)
ROW_RE = re.compile(
    r"^\s{2}(\S+)\s+median\s+([\d.]+)\s+ms"
)


def parse_benchmark_text(text: str) -> list[dict]:
    rows: list[dict] = []
    current: dict | None = None
    for line in text.splitlines():
        m_sec = SECTION_RE.match(line.strip())
        if m_sec:
            current = {
                "N": int(m_sec.group(1)),
                "density": float(m_sec.group(2)),
                "regime": m_sec.group(3),
            }
            continue
        m_row = ROW_RE.match(line)
        if m_row and current is not None:
            method = m_row.group(1)
            if method in EXCLUDE_METHODS:
                continue
            rows.append(
                {
                    **current,
                    "method": method,
                    "median_ms": float(m_row.group(2)),
                }
            )
    return rows


def lookup(rows: list[dict], regime: str, N: int, method: str) -> float:
    for r in rows:
        if r["regime"] == regime and r["N"] == N and r["method"] == method:
            return r["median_ms"]
    raise KeyError(f"missing: regime={regime} N={N} method={method}")


def plot_benchmark(
    rows: list[dict],
    output: Path,
    dpi: int,
    title: str,
    methods: list[str],
) -> None:
    n_methods = len(methods)
    palette = plt.cm.tab10(np.linspace(0, 1, max(n_methods, 10)))
    method_colors = {m: palette[k] for k, m in enumerate(methods)}

    bar_w = 0.09
    group_gap = 0.55
    group_pitch = n_methods * bar_w + group_gap
    group_center = (n_methods - 1) * bar_w / 2

    fig, axes = plt.subplots(
        len(DENSITY_SCENARIOS),
        1,
        figsize=(11, 3.6 * len(DENSITY_SCENARIOS)),
        sharex=False,
    )
    if len(DENSITY_SCENARIOS) == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=13, y=0.995)

    for ax, (density, label) in zip(axes, DENSITY_SCENARIOS):
        for i, N in enumerate(SIZES):
            x0 = i * group_pitch
            for k, m in enumerate(methods):
                y = lookup(rows, label, N, m)
                ax.bar(x0 + k * bar_w, y, width=bar_w, color=method_colors[m])
        tick_x = [i * group_pitch + group_center for i in range(len(SIZES))]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([str(n) for n in SIZES], fontsize=11)
        ax.set_xlim(-bar_w, len(SIZES) * group_pitch - group_gap + n_methods * bar_w)
        ax.set_ylabel("median ms")
        ax.set_title(f"{label} (density={density})")
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("matrix size N")
    legend_patches = [Patch(facecolor=method_colors[m], label=m) for m in methods]
    fig.legend(
        handles=legend_patches,
        fontsize=6,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.subplots_adjust(top=0.90, bottom=0.10, hspace=0.35)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output.resolve()}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot HIP benchmark bar chart")
    parser.add_argument(
        "--input",
        type=Path,
        help="Parse benchmark stdout from file (else use embedded RX 6800 XT data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hip_benchmark_rx6800xt.png"),
        help="Output PNG path",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--title",
        default="HIP benchmark — AMD Radeon RX 6800 XT (gfx1030)",
    )
    args = parser.parse_args()

    if args.input:
        text = args.input.read_text(encoding="utf-8")
        rows = parse_benchmark_text(text)
        if not rows:
            print("No rows parsed from input.", file=sys.stderr)
            return 1
    else:
        rows = DEFAULT_ROWS

    methods = [m for m in METHODS if any(r["method"] == m for r in rows)]
    plot_benchmark(rows, args.output, args.dpi, args.title, methods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
