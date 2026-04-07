"""
report.py — generate a Markdown/HTML summary report from benchmark results.

Usage:
    python3 python/report.py results.json --gpu a100 --out report.md
    python3 python/report.py results.json --html --out report.html
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "tools" / "roofline"))
from hw_limits import A100_SXM4_80GB, H100_SXM5_80GB

GPU_SPECS = {"a100": A100_SXM4_80GB, "h100": H100_SXM5_80GB}

PHASE_ORDER = ["naive", "shmem", "swizzle", "wmma", "pipeline", "ptx", "hopper"]


def load_results(path: str) -> pd.DataFrame:
    return pd.read_json(path)


def generate_markdown(df: pd.DataFrame, gpu: str, dtype: str) -> str:
    spec = GPU_SPECS[gpu]
    lines = []
    lines.append(f"# GEMM Benchmark Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"GPU: **{spec.name}** ({spec.sm_arch}) | dtype: **{dtype.upper()}**")
    lines.append("")

    # ── cuBLAS baseline ───────────────────────────────────────────────────────
    lines.append("## cuBLAS Baseline (TFLOPS)")
    lines.append("")

    # ── Per-phase best table ───────────────────────────────────────────────────
    lines.append("## Best TFLOPS per Phase")
    lines.append("")
    lines.append("| Phase | Best TFLOPS | % of cuBLAS Peak | Best Tile |")
    lines.append("|-------|-------------|------------------|-----------|")

    sub = df[(df.dtype == dtype) & (df.get("error", "").fillna("") == "")]
    best = sub.loc[sub.groupby("phase")["tflops"].idxmax()].set_index("phase")

    for phase in PHASE_ORDER:
        if phase not in best.index:
            continue
        row  = best.loc[phase]
        tile = f"BM={int(row.BM)} BN={int(row.BN)} BK={int(row.BK)}"
        pct  = row.get("pct_peak", row.get("pct_cublas_peak", 0.0))
        lines.append(
            f"| {phase} | {row.tflops:.2f} | {pct:.1f}% | {tile} |"
        )
    lines.append("")

    # ── Per-size table for largest square problem ─────────────────────────────
    sq = sub[(sub.M == sub.N) & (sub.N == sub.K)]
    if not sq.empty:
        max_size = sq.M.max()
        sq_best  = sq[sq.M == max_size].loc[sq[sq.M == max_size].groupby("phase")["tflops"].idxmax()]
        lines.append(f"## {max_size}×{max_size}×{max_size} by Phase")
        lines.append("")
        lines.append("| Phase | TFLOPS | % Peak | mean_ms |")
        lines.append("|-------|--------|--------|---------|")
        for phase in PHASE_ORDER:
            row = sq_best[sq_best.phase == phase]
            if row.empty:
                continue
            r   = row.iloc[0]
            pct = r.get("pct_peak", r.get("pct_cublas_peak", 0.0))
            lines.append(
                f"| {phase} | {r.tflops:.2f} | {pct:.1f}% | {r.mean_ms:.3f} |"
            )
        lines.append("")

    # ── Interpretation notes ──────────────────────────────────────────────────
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Compute peak (FP16 tensor): **{spec.fp16_tensor_tflops:.0f} TFLOPS**")
    lines.append(f"- Memory bandwidth: **{spec.hbm_bandwidth_gbs:.0f} GB/s**")
    lines.append(f"- L2 cache: **{spec.l2_cache_mb:.0f} MB**")
    lines.append(f"- Shared mem per SM: **{spec.smem_per_sm_kb:.0f} KB**")
    lines.append("")
    lines.append("See `tools/roofline/roofline.py` for the roofline plot.")
    lines.append("See `tools/sass/` for SASS analysis.")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Benchmark results JSON")
    parser.add_argument("--gpu",   default="a100", choices=["a100", "h100"])
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--out",   default="report.md")
    parser.add_argument("--html",  action="store_true",
                        help="Wrap markdown in basic HTML")
    args = parser.parse_args()

    df   = load_results(args.json_file)
    md   = generate_markdown(df, args.gpu, args.dtype)

    if args.html:
        try:
            import markdown as md_lib
            html = f"<html><body>{md_lib.markdown(md, extensions=['tables'])}</body></html>"
            content = html
        except ImportError:
            print("markdown package not installed; falling back to .md output")
            content = md
            args.out = args.out.replace(".html", ".md")
    else:
        content = md

    with open(args.out, "w") as f:
        f.write(content)
    print(f"Report → {args.out}")


if __name__ == "__main__":
    main()
