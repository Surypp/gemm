"""
plot.py — four plot types for GEMM benchmark results.

1. phase_bar      : TFLOPS per phase for a fixed (M, N, K, dtype)
2. tile_heatmap   : TFLOPS as a function of BM × BK for a fixed phase
3. size_scaling   : TFLOPS vs M (square problems) for selected phases
4. roofline       : scatter (arithmetic_intensity, TFLOPS) with hardware roof

Usage:
    from python.plot import phase_bar, tile_heatmap, size_scaling, roofline_plot
    import pandas as pd
    df = pd.read_json("results.json")
    phase_bar(df, M=4096, N=4096, K=4096, dtype="fp16")
"""
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "roofline"))
from hw_limits import (A100_SXM4_80GB, H100_SXM5_80GB,
                       arithmetic_intensity_gemm, roofline_peak)

GPU_SPECS = {"a100": A100_SXM4_80GB, "h100": H100_SXM5_80GB}

PHASE_ORDER  = ["naive", "shmem", "swizzle", "wmma", "pipeline", "ptx", "hopper"]
PHASE_COLORS = {
    "naive":    "#e74c3c",
    "shmem":    "#e67e22",
    "swizzle":  "#f1c40f",
    "wmma":     "#2ecc71",
    "pipeline": "#3498db",
    "ptx":      "#9b59b6",
    "hopper":   "#1abc9c",
}


def _load(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source
    return pd.read_json(source)


# ── 1. Phase progression bar chart ───────────────────────────────────────────

def phase_bar(
    data,
    M: int, N: int, K: int,
    dtype: str = "fp16",
    best_tile: bool = True,
    ax=None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Bar chart: TFLOPS for each phase on a single (M, N, K, dtype) problem.
    If best_tile=True, picks the tile config with highest TFLOPS per phase.
    """
    df = _load(data)
    mask = (df.M == M) & (df.N == N) & (df.K == K) & (df.dtype == dtype)
    sub  = df[mask].copy()

    if best_tile:
        sub = sub.loc[sub.groupby("phase")["tflops"].idxmax()]
    else:
        sub = sub.groupby("phase", as_index=False)["tflops"].mean()

    # Filter out error rows
    sub = sub[sub.get("error", "").fillna("") == ""]
    sub["phase"] = pd.Categorical(sub["phase"], categories=PHASE_ORDER, ordered=True)
    sub = sub.sort_values("phase")

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    colors = [PHASE_COLORS.get(p, "#95a5a6") for p in sub["phase"]]
    bars = ax.bar(sub["phase"], sub["tflops"], color=colors, edgecolor="white", linewidth=0.8)

    # Annotate with % of cuBLAS if available
    if "pct_peak" in sub.columns:
        for bar, (_, row) in zip(bars, sub.iterrows()):
            pct = row.get("pct_peak", 0)
            if pct > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.0f}%",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Phase", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(title or f"GEMM {M}×{N}×{K}  [{dtype.upper()}]", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


# ── 2. Tile sweep heatmap ─────────────────────────────────────────────────────

def tile_heatmap(
    data,
    phase: str,
    M: int, N: int, K: int,
    dtype: str = "fp16",
    ax=None,
) -> plt.Axes:
    """
    Heatmap: TFLOPS as function of BM (rows) × BK (columns) for a fixed phase.
    """
    df  = _load(data)
    sub = df[(df.phase == phase) & (df.M == M) & (df.N == N) &
             (df.K == K) & (df.dtype == dtype)]
    sub = sub[sub.get("error", "").fillna("") == ""]

    if sub.empty:
        print(f"No data for phase={phase} {M}×{N}×{K} {dtype}")
        return ax

    pivot = sub.pivot_table(index="BM", columns="BK", values="tflops", aggfunc="max")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("BK", fontsize=11)
    ax.set_ylabel("BM", fontsize=11)
    ax.set_title(f"{phase}  [{dtype}]  {M}×{N}×{K}", fontsize=12)
    plt.colorbar(im, ax=ax, label="TFLOPS")

    # Annotate cells
    for i, bm in enumerate(pivot.index):
        for j, bk in enumerate(pivot.columns):
            val = pivot.loc[bm, bk]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color="black")
    plt.tight_layout()
    return ax


# ── 3. Size scaling ───────────────────────────────────────────────────────────

def size_scaling(
    data,
    phases: Optional[list[str]] = None,
    dtype: str = "fp16",
    ax=None,
) -> plt.Axes:
    """
    Line plot: TFLOPS vs square matrix size for selected phases.
    Picks the best tile per (phase, size).
    """
    df = _load(data)
    if phases is None:
        phases = [p for p in PHASE_ORDER if p in df.phase.values]

    # Square problems only
    sub = df[(df.M == df.N) & (df.N == df.K) & (df.dtype == dtype)]
    sub = sub[sub.get("error", "").fillna("") == ""]
    sub = sub.loc[sub.groupby(["phase", "M"])["tflops"].idxmax()]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    for phase in phases:
        ps = sub[sub.phase == phase].sort_values("M")
        if ps.empty:
            continue
        color = PHASE_COLORS.get(phase, "#95a5a6")
        ax.semilogx(ps.M, ps.tflops, "o-", color=color, label=phase, linewidth=2)

    ax.set_xlabel("Matrix size (M=N=K)", fontsize=11)
    ax.set_ylabel("TFLOPS", fontsize=11)
    ax.set_title(f"Performance vs Matrix Size  [{dtype.upper()}]", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    return ax


# ── 4. Roofline overlay ───────────────────────────────────────────────────────

def roofline_plot(
    data,
    gpu: str = "a100",
    dtype: str = "fp16",
    phases: Optional[list[str]] = None,
    ax=None,
    out: Optional[str] = None,
) -> plt.Axes:
    """
    Scatter plot of (arithmetic_intensity, TFLOPS) with hardware roofline.
    """
    df = _load(data)
    spec = GPU_SPECS[gpu]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    # Roofline curve
    ai_range = np.logspace(-1, 3, 500)
    roof = np.array([roofline_peak(spec, ai, dtype) for ai in ai_range])
    ax.loglog(ai_range, roof, "k-", linewidth=2.5, label="Roofline")

    compute_peak = getattr(spec, f"{dtype}_tensor_tflops", spec.fp16_tensor_tflops)
    ridge_ai     = compute_peak * 1e3 / spec.hbm_bandwidth_gbs
    ax.axvline(ridge_ai, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.axhline(compute_peak, color="gray", linestyle=":", alpha=0.5, linewidth=1)

    # Data points
    if phases is None:
        phases = [p for p in PHASE_ORDER if p in df.phase.values]

    sub = df[(df.dtype == dtype) & df.phase.isin(phases)]
    sub = sub[sub.get("error", "").fillna("") == ""]
    sub = sub.loc[sub.groupby(["phase", "M", "N", "K"])["tflops"].idxmax()]

    for phase in phases:
        ps    = sub[sub.phase == phase]
        color = PHASE_COLORS.get(phase, "#95a5a6")
        for _, row in ps.iterrows():
            ai     = arithmetic_intensity_gemm(int(row.M), int(row.N), int(row.K),
                                               dtype_bytes=2 if dtype == "fp16" else 4)
            tflops = row["tflops"]
            ax.scatter(ai, tflops, color=color, s=80, zorder=5, alpha=0.85)

    legend_items = [mpatches.Patch(color="k", label="Roofline")] + [
        mpatches.Patch(color=PHASE_COLORS.get(p, "#95a5a6"), label=p)
        for p in phases
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=9)
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)", fontsize=11)
    ax.set_ylabel("Performance (TFLOPS)", fontsize=11)
    ax.set_title(f"Roofline — {spec.name}  [{dtype.upper()}]", fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    if out:
        plt.savefig(out, dpi=150)
        print(f"Saved → {out}")
    return ax
