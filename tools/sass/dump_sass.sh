#!/usr/bin/env bash
# Dump PTX and SASS from a CUDA binary, optionally filtering by kernel name.
#
# Usage:
#   ./tools/sass/dump_sass.sh <binary> [kernel_filter]
#
# Example:
#   ./tools/sass/dump_sass.sh build/bench/gemm_bench gemm_wmma_kernel
#   # Dumps all kernels if no filter given

set -euo pipefail

BINARY="${1:?Usage: $0 <binary> [kernel_filter]}"
FILTER="${2:-}"
OUTDIR="sass_dumps"
mkdir -p "$OUTDIR"

BASE=$(basename "$BINARY")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

PTX_FILE="$OUTDIR/${BASE}_${TIMESTAMP}.ptx"
SASS_FILE="$OUTDIR/${BASE}_${TIMESTAMP}.sass"

echo "Binary : $BINARY"
echo "PTX  → $PTX_FILE"
echo "SASS → $SASS_FILE"

cuobjdump --dump-ptx  "$BINARY" > "$PTX_FILE"
cuobjdump --dump-sass "$BINARY" > "$SASS_FILE"

# Filter to a specific kernel if requested
if [[ -n "$FILTER" ]]; then
    FILTERED_SASS="$OUTDIR/${BASE}_${FILTER}_${TIMESTAMP}.sass"
    # Extract section between "Function : $FILTER" and next "Function :" line
    awk "/Function : .*${FILTER}/ { found=1 }
         found && /Function : / && !/Function : .*${FILTER}/ { found=0 }
         found { print }" "$SASS_FILE" > "$FILTERED_SASS"
    echo "Filtered SASS (${FILTER}) → $FILTERED_SASS"
    LINES=$(wc -l < "$FILTERED_SASS")
    echo "  $LINES lines"
fi

# Count HMMA (tensor core) instructions
HMMA_COUNT=$(grep -c "HMMA" "$SASS_FILE" 2>/dev/null || echo 0)
echo ""
echo "HMMA instruction count in full binary: $HMMA_COUNT"
echo ""
echo "To compare with cuBLAS:"
echo "  1. Find the cuBLAS .so: ldd your_binary | grep cublas"
echo "  2. cuobjdump --dump-sass /path/to/libcublas.so.X > cublas.sass"
echo "  3. grep 'HMMA' cublas.sass | wc -l"
