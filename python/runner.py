"""
runner.py — subprocess wrapper for the C++ bench binary.

The binary is invoked with structured arguments and writes JSON to a temp file.
This module reads that JSON and returns a list of result dicts.
"""
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


DEFAULT_BINARY = str(Path(__file__).parent.parent / "build" / "bench" / "gemm_bench")


class BenchError(RuntimeError):
    pass


def run_bench(
    binary: str = DEFAULT_BINARY,
    dtype: str = "fp16",
    warmup: int = 5,
    iters: int = 20,
    device: int = 0,
    extra_args: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Run the benchmark binary and return parsed JSON results.

    Returns a list of row dicts (one per phase × problem × tile combination).
    Rows with errors have a non-empty 'error' key.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = tmp.name

    cmd = [
        binary,
        "--dtype",   dtype,
        "--warmup",  str(warmup),
        "--iters",   str(iters),
        "--device",  str(device),
        "--json",    json_path,
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise BenchError(
            f"Bench binary failed (exit {e.returncode}):\n"
            f"stdout: {e.stdout[-2000:]}\n"
            f"stderr: {e.stderr[-2000:]}"
        ) from e
    except FileNotFoundError:
        raise BenchError(
            f"Bench binary not found: {binary}\n"
            "Build with: cmake -B build && cmake --build build"
        )

    try:
        with open(json_path) as f:
            rows = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise BenchError(f"Failed to read JSON output: {e}") from e

    # Print stdout (contains per-phase progress lines)
    if result.stdout:
        print(result.stdout, end="")

    return rows
