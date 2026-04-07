#!/usr/bin/env bash
# Nsight Compute profiling presets for GEMM kernels.
# Usage: ./tools/profile/ncu_profile.sh <preset> <binary> [binary_args...]
#
# Presets:
#   summary     — quick 3-metric pass, ~2× kernel replay
#   memory      — memory bandwidth and bank conflicts
#   tensor      — tensor core utilization
#   occupancy   — warp occupancy and register pressure
#   full        — all metrics (slow, many replays)
#   sass        — dump PTX + SASS to files (no metrics)

set -euo pipefail

PRESET="${1:-summary}"
shift
BINARY="${1:?Usage: $0 <preset> <binary> [args...]}"
shift
ARGS="${*:-}"

# Output directory
OUTDIR="ncu_reports"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

NCU_BIN="ncu"
command -v ncu &>/dev/null || { echo "ERROR: ncu not found in PATH"; exit 1; }

case "$PRESET" in

summary)
    METRICS="sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
lts__t_sector_hit_rate.pct"
    REPORT="$OUTDIR/summary_${TIMESTAMP}.ncu-rep"
    echo "Running SUMMARY profile → $REPORT"
    $NCU_BIN \
        --metrics "$METRICS" \
        --export "$REPORT" \
        --force-overwrite \
        "$BINARY" $ARGS
    ;;

memory)
    METRICS="l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
lts__t_bytes_equiv_l1sectmiss_pipe_lsu_mem_global_op_ld.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
    REPORT="$OUTDIR/memory_${TIMESTAMP}.ncu-rep"
    echo "Running MEMORY profile → $REPORT"
    $NCU_BIN --metrics "$METRICS" --export "$REPORT" --force-overwrite "$BINARY" $ARGS
    ;;

tensor)
    METRICS="sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_tensor.sum,\
sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
    REPORT="$OUTDIR/tensor_${TIMESTAMP}.ncu-rep"
    echo "Running TENSOR CORE profile → $REPORT"
    $NCU_BIN --metrics "$METRICS" --export "$REPORT" --force-overwrite "$BINARY" $ARGS
    ;;

occupancy)
    METRICS="sm__maximum_warps_per_active_cycle_pct,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warp_latency_due_to_long_scoreboard.ratio,\
smsp__average_warp_latency_due_to_memory_dependency_pipe_lsu.ratio"
    REPORT="$OUTDIR/occupancy_${TIMESTAMP}.ncu-rep"
    echo "Running OCCUPANCY profile → $REPORT"
    $NCU_BIN --metrics "$METRICS" --export "$REPORT" --force-overwrite "$BINARY" $ARGS
    ;;

full)
    REPORT="$OUTDIR/full_${TIMESTAMP}.ncu-rep"
    echo "Running FULL profile (slow) → $REPORT"
    $NCU_BIN \
        --set full \
        --export "$REPORT" \
        --force-overwrite \
        "$BINARY" $ARGS
    ;;

sass)
    echo "Dumping PTX + SASS for $BINARY"
    cuobjdump --dump-ptx  "$BINARY" > "$OUTDIR/dump_${TIMESTAMP}.ptx"
    cuobjdump --dump-sass "$BINARY" > "$OUTDIR/dump_${TIMESTAMP}.sass"
    echo "PTX  → $OUTDIR/dump_${TIMESTAMP}.ptx"
    echo "SASS → $OUTDIR/dump_${TIMESTAMP}.sass"
    ;;

*)
    echo "Unknown preset: $PRESET"
    echo "Valid presets: summary memory tensor occupancy full sass"
    exit 1
    ;;
esac

echo "Done."
