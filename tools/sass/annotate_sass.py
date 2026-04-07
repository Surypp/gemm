#!/usr/bin/env python3
"""
Annotate SASS output with instruction category labels.

Usage:
    python3 tools/sass/annotate_sass.py dump.sass [--kernel gemm_wmma_kernel]

Output: instruction frequency table + annotated SASS with categories.

Categories:
    HMMA      — tensor core multiply-accumulate
    LDG/STG   — global (HBM) load/store
    LDS/STS   — shared memory load/store
    IMAD/IADD — integer (addressing) arithmetic
    FFMA/FADD — FP32 scalar ops
    BAR/SYNC  — synchronization barriers
    S2R       — special register reads (lane_id, warp_id, etc.)
    OTHER     — everything else
"""
import re
import sys
import argparse
from collections import Counter


CATEGORIES = {
    "HMMA":  re.compile(r"\bHMMA\b"),
    "LDG":   re.compile(r"\bLDG\b"),
    "STG":   re.compile(r"\bSTG\b"),
    "LDS":   re.compile(r"\bLDS\b"),
    "STS":   re.compile(r"\bSTS\b"),
    "FFMA":  re.compile(r"\bFFMA\b"),
    "FADD":  re.compile(r"\bFADD\b"),
    "IMAD":  re.compile(r"\bIMAD\b"),
    "IADD":  re.compile(r"\bIADD\b"),
    "BAR":   re.compile(r"\bBAR\b"),
    "BSYNC": re.compile(r"\bBSYNC\b"),
    "S2R":   re.compile(r"\bS2R\b"),
    "MEMBAR":re.compile(r"\bMEMBAR\b"),
}


def categorize(instr_line: str) -> str:
    for cat, pattern in CATEGORIES.items():
        if pattern.search(instr_line):
            return cat
    return "OTHER"


def parse_kernel_sections(sass_text: str, kernel_filter: str | None) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in sass_text.splitlines():
        m = re.match(r"\s*Function\s*:\s*(.+)", line)
        if m:
            current = m.group(1).strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)

    if kernel_filter:
        sections = {k: v for k, v in sections.items() if kernel_filter in k}
    return sections


def analyze_section(lines: list[str]) -> tuple[Counter, list[tuple[str, str]]]:
    counts: Counter = Counter()
    annotated: list[tuple[str, str]] = []
    for line in lines:
        # Only process instruction lines (they contain /* offset */ opcode)
        if re.search(r"/\*\s*0x[0-9a-f]+\s*\*/", line):
            cat = categorize(line)
            counts[cat] += 1
            annotated.append((cat, line))
        else:
            annotated.append(("", line))
    return counts, annotated


def print_frequency_table(kernel_name: str, counts: Counter, total: int):
    print(f"\n{'='*60}")
    print(f"Kernel: {kernel_name}")
    print(f"{'='*60}")
    print(f"{'Category':<12} {'Count':>8} {'%':>7}")
    print("-" * 30)
    for cat, n in counts.most_common():
        pct = 100.0 * n / total if total else 0.0
        marker = " ← tensor core" if cat == "HMMA" else ""
        print(f"  {cat:<10} {n:>8} {pct:>6.1f}%{marker}")
    print(f"  {'TOTAL':<10} {total:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sass_file", help="SASS dump file from cuobjdump")
    parser.add_argument("--kernel", help="Filter to kernel containing this string")
    parser.add_argument("--annotate", action="store_true",
                        help="Print annotated SASS with category labels")
    args = parser.parse_args()

    with open(args.sass_file) as f:
        text = f.read()

    sections = parse_kernel_sections(text, args.kernel)
    if not sections:
        print(f"No kernels found" + (f" matching '{args.kernel}'" if args.kernel else ""))
        sys.exit(1)

    for kernel_name, lines in sections.items():
        counts, annotated = analyze_section(lines)
        total = sum(counts.values())
        print_frequency_table(kernel_name, counts, total)

        if args.annotate:
            print("\n--- Annotated SASS ---")
            for cat, line in annotated:
                prefix = f"[{cat:<6}] " if cat else "         "
                print(prefix + line)


if __name__ == "__main__":
    main()
