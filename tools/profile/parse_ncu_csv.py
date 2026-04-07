#!/usr/bin/env python3
"""
Parse an Nsight Compute CSV export and print a clean summary.

Usage:
    ncu --csv --metrics ... ./gemm_bench ... > raw.csv
    python3 tools/profile/parse_ncu_csv.py raw.csv [--sort tflops]
"""
import csv
import sys
import argparse
from collections import defaultdict


INTERESTING_METRICS = {
    "sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active":
        "Tensor util %",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum":
        "Bank conflicts (ld)",
    "sm__warps_active.avg.pct_of_peak_sustained_active":
        "Warp occupancy %",
    "lts__t_sector_hit_rate.pct":
        "L2 hit rate %",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum":
        "HBM bytes read",
    "sm__maximum_warps_per_active_cycle_pct":
        "Theoretical occupancy %",
}


def parse_ncu_csv(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_metrics(rows: list[dict]) -> dict:
    """Group rows by kernel name, collect metric values."""
    kernels: dict[str, dict] = defaultdict(dict)
    for row in rows:
        kernel = row.get("Kernel Name", row.get("ID", "unknown"))
        metric = row.get("Metric Name", "")
        value  = row.get("Metric Value", "")
        unit   = row.get("Metric Unit", "")
        if metric in INTERESTING_METRICS:
            try:
                kernels[kernel][metric] = float(value.replace(",", ""))
            except ValueError:
                kernels[kernel][metric] = value
    return kernels


def print_summary(kernels: dict):
    for kernel_name, metrics in kernels.items():
        print(f"\nKernel: {kernel_name}")
        print("-" * 60)
        for metric_key, display_name in INTERESTING_METRICS.items():
            val = metrics.get(metric_key, "—")
            if isinstance(val, float):
                print(f"  {display_name:<30} {val:.2f}")
            else:
                print(f"  {display_name:<30} {val}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", help="NCU CSV export")
    args = parser.parse_args()

    rows    = parse_ncu_csv(args.csv_file)
    kernels = extract_metrics(rows)
    if not kernels:
        print(f"No recognized metrics found in {args.csv_file}")
        print("Make sure you exported with: ncu --csv --metrics <metric_list>")
        sys.exit(1)
    print_summary(kernels)


if __name__ == "__main__":
    main()
