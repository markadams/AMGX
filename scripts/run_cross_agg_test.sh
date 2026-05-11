#!/bin/bash
# =============================================================================
# run_cross_agg_test.sh
#
# Cross-aggregate test: GAMG vs AMGx on 100x100 Poisson (Step 6 of plan).
#
# Purpose:
#   Isolate whether the convergence gap between PETSc GAMG (~16 iters) and
#   AMGx SA-MIS2 (~82 iters) is due to aggregate quality or solver machinery.
#
# 4-combination test matrix:
#   A. PETSc solver + GAMG aggregates  (PETSc baseline)
#   B. AMGx solver  + GAMG aggregates  (GAMG aggs, AMGx machinery)
#   C. AMGx solver  + AMGx aggregates  (AMGx baseline)
#   D. PETSc solver + AMGx aggregates  (AMGx aggs, PETSc machinery)
#
# Interpretation:
#   If B ≈ A (and D ≈ C): aggregate quality is the sole factor
#   If B >> A (and D << C): AMGx solver machinery also contributes
#   If B ≈ C (and D ≈ A): solver machinery is the main factor
#
# Prerequisites:
#   - AMGx built in $AMGX_BUILD_DIR (with export/import hooks compiled in)
#   - PETSc built in $PETSC_DIR with GAMG export/import hooks compiled in
#   - test_cross_agg binary built in $AMGX_BUILD_DIR
#   - PETSc ex2 binary available at $PETSC_EX2
#
# Run on Perlmutter (interactive GPU node):
#   salloc -N 1 -C gpu -q interactive -t 01:00:00 -A <account>
#   bash ~/Codes/amgx/scripts/run_cross_agg_test.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — adjust these paths for your Perlmutter environment
# ---------------------------------------------------------------------------
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$HOME/amgx-sa/build_perlmutter}"
PETSC_DIR="${PETSC_DIR:-$HOME/Codes/petsc}"
PETSC_EX2="${PETSC_EX2:-$PETSC_DIR/src/ksp/ksp/tutorials/ex2}"
AGG_DIR="${AGG_DIR:-/tmp/cross_agg_$$}"

# Perlmutter CUDA/math library paths (adjust if SDK version changes)
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$MATH_LIB:${LD_LIBRARY_PATH:-}"

MATRIX_FILE="$AGG_DIR/poisson2d_100.mtx"
GAMG_AGG_FILE="$AGG_DIR/gamg_aggregates.txt"
AMGX_AGG_FILE="$AGG_DIR/amgx_aggregates.txt"

AMGX_CROSS="$AMGX_BUILD_DIR/test_cross_agg"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
if [ ! -f "$AMGX_CROSS" ]; then
    echo "ERROR: test_cross_agg not found at $AMGX_CROSS"
    echo "Build it from $AMGX_BUILD_DIR with:"
    echo "  /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \\"
    echo "    ../examples/test_cross_agg.c -o test_cross_agg \\"
    echo "    ./libamgxsh.so -lrt -ldl \\"
    echo "    $CUDA_LIB/libcudart_static.a \\"
    echo "    $MATH_LIB/libcublas.so $MATH_LIB/libcusolver.so $MATH_LIB/libcusparse.so \\"
    echo "    $CUDA_LIB/libculibos.a $CUDA_LIB/libnvJitLink.so $MATH_LIB/libcublasLt.so \\"
    echo "    -lm -lpthread -ldl"
    exit 1
fi

if [ ! -f "$PETSC_EX2" ]; then
    echo "WARNING: PETSc ex2 not found at $PETSC_EX2"
    echo "Tests A and D (PETSc solver) will be skipped."
    SKIP_PETSC=1
else
    SKIP_PETSC=0
fi

mkdir -p "$AGG_DIR"
echo "Aggregate files will be stored in: $AGG_DIR"

# ---------------------------------------------------------------------------
# Step 0: Generate 100x100 Poisson matrix
# ---------------------------------------------------------------------------
echo ""
echo "=== Step 0: Generate 100x100 Poisson matrix ==="
cd "$AMGX_BUILD_DIR"
./examples/generate_poisson -p 5 100 100 -o "$MATRIX_FILE"
echo "Matrix written to: $MATRIX_FILE"

# ---------------------------------------------------------------------------
# Test A: PETSc solver + GAMG aggregates (PETSc baseline)
#         Also exports GAMG aggregates for Test B.
# ---------------------------------------------------------------------------
echo ""
echo "=== Test A: PETSc solver + GAMG aggregates (PETSc baseline) ==="
if [ "$SKIP_PETSC" -eq 1 ]; then
    echo "SKIPPED (PETSc ex2 not found)"
else
    PETSC_EXPORT_AGGREGATES="$GAMG_AGG_FILE" \
    "$PETSC_EX2" \
        -da_grid_x 100 -da_grid_y 100 \
        -pc_type gamg \
        -pc_gamg_type agg \
        -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_mat_coarsen_type misk \
        -pc_gamg_mat_coarsen_misk_distance 2 \
        -pc_gamg_levels 2 \
        -mg_levels_ksp_max_it 2 \
        -mg_levels_ksp_type chebyshev \
        -mg_coarse_pc_type lu \
        -ksp_monitor \
        -ksp_converged_reason \
        -ksp_rtol 1e-5 \
        2>&1 | tee "$AGG_DIR/test_A_petsc_gamg_aggs.log"
    echo "GAMG aggregates exported to: $GAMG_AGG_FILE"
fi

# ---------------------------------------------------------------------------
# Test C: AMGx solver + AMGx MIS-2 aggregates (AMGx baseline)
#         Also exports AMGx aggregates for Test D.
# ---------------------------------------------------------------------------
echo ""
echo "=== Test C: AMGx solver + AMGx MIS-2 aggregates (AMGx baseline) ==="
cd "$AMGX_BUILD_DIR"
AMGX_EXPORT_AGGREGATES="$AMGX_AGG_FILE" \
"$AMGX_CROSS" "$MATRIX_FILE" \
    2>&1 | tee "$AGG_DIR/test_C_amgx_amgx_aggs.log"
echo "AMGx aggregates exported to: $AMGX_AGG_FILE"

# ---------------------------------------------------------------------------
# Test B: AMGx solver + GAMG aggregates
#         (Tests whether GAMG's better aggregates improve AMGx convergence)
# ---------------------------------------------------------------------------
echo ""
echo "=== Test B: AMGx solver + GAMG aggregates ==="
if [ "$SKIP_PETSC" -eq 1 ] || [ ! -f "$GAMG_AGG_FILE" ]; then
    echo "SKIPPED (GAMG aggregate file not available: $GAMG_AGG_FILE)"
else
    cd "$AMGX_BUILD_DIR"
    AMGX_IMPORT_AGGREGATES="$GAMG_AGG_FILE" \
    "$AMGX_CROSS" "$MATRIX_FILE" \
        2>&1 | tee "$AGG_DIR/test_B_amgx_gamg_aggs.log"
fi

# ---------------------------------------------------------------------------
# Test D: PETSc solver + AMGx aggregates
#         (Tests whether AMGx's aggregates hurt PETSc convergence)
# ---------------------------------------------------------------------------
echo ""
echo "=== Test D: PETSc solver + AMGx aggregates ==="
if [ "$SKIP_PETSC" -eq 1 ]; then
    echo "SKIPPED (PETSc ex2 not found)"
elif [ ! -f "$AMGX_AGG_FILE" ]; then
    echo "SKIPPED (AMGx aggregate file not available: $AMGX_AGG_FILE)"
else
    PETSC_IMPORT_AGGREGATES="$AMGX_AGG_FILE" \
    "$PETSC_EX2" \
        -da_grid_x 100 -da_grid_y 100 \
        -pc_type gamg \
        -pc_gamg_type agg \
        -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_levels 2 \
        -mg_levels_ksp_max_it 2 \
        -mg_levels_ksp_type chebyshev \
        -mg_coarse_pc_type lu \
        -ksp_monitor \
        -ksp_converged_reason \
        -ksp_rtol 1e-5 \
        2>&1 | tee "$AGG_DIR/test_D_petsc_amgx_aggs.log"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary ==="
echo "Log files in: $AGG_DIR"
echo ""
echo "Iteration counts (grep from logs):"
for log in "$AGG_DIR"/test_*.log; do
    [ -f "$log" ] || continue
    label=$(basename "$log" .log)
    # PETSc prints "Linear solve converged due to ... iterations N"
    # AMGx prints "Iterations: N" or similar in print_solve_stats
    iters_petsc=$(grep -oP 'iterations \K[0-9]+' "$log" 2>/dev/null | tail -1 || true)
    iters_amgx=$(grep -oP 'Iterations:\s*\K[0-9]+' "$log" 2>/dev/null | tail -1 || true)
    iters="${iters_petsc:-${iters_amgx:-?}}"
    echo "  $label: $iters iterations"
done

echo ""
echo "Interpretation:"
echo "  If Test B iters ≈ Test A iters: GAMG aggs help AMGx → aggregate quality is key"
echo "  If Test B iters >> Test A iters: AMGx solver machinery also contributes"
echo "  If Test D iters ≈ Test C iters: AMGx aggs hurt PETSc → aggregate quality is key"
echo "  If Test D iters ≈ Test A iters: PETSc machinery compensates for bad aggs"
