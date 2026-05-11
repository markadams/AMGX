#!/bin/bash
# =============================================================================
# run_jacobi_l1_test.sh
#
# §7.9 Jacobi-L1 Richardson Smoother Experiment
#
# Goal: Freeze aggregates (use GAMG's) and use an identical, simple smoother
# (undamped Jacobi-L1 = Richardson + L1-row-norm diagonal) so the only
# remaining variable is the SA prolongator construction machinery.
#
# Three tests:
#   A. PETSc GAMG + GAMG aggregates + Jacobi-L1 Richardson smoother (PETSc baseline)
#   B. AMGx SA   + GAMG aggregates + Jacobi-L1 Richardson smoother (AMGx Run 4)
#   C. AMGx SA   + AMGx MIS-2 aggs + Jacobi-L1 Richardson smoother (AMGx Run 4)
#
# Smoother definition (identical in both solvers):
#   x_{k+1} = x_k + D_L1^{-1} (b - A x_k)
#   (D_L1)_ii = sum_j |A_ij|   (L1 row norm, including diagonal)
#   omega = 1 (undamped Richardson)
#
# PETSc options:
#   -mg_levels_ksp_type richardson
#   -mg_levels_ksp_max_it 1
#   -mg_levels_pc_type jacobi
#   -mg_levels_pc_jacobi_type rowl1
#   -mg_levels_pc_jacobi_rowl1_scale 1
#   -mg_levels_pc_jacobi_fixdiagonal
#
# AMGx config: JACOBI_L1 solver, relaxation_factor=1.0 (Run 4 in test_cross_agg)
#
# Interpretation:
#   If B iters ≈ A iters: SA prolongator construction is equivalent to GAMG's
#   If B iters > A iters: gap is in prolongator construction (P_tent QR, P_smooth omega)
#   C vs B: effect of aggregate quality with identical smoother and machinery
#
# Prerequisites:
#   - AMGx built in $AMGX_BUILD_DIR (libamgxsh.so + test_cross_agg binary)
#   - PETSc built in $PETSC_DIR with GAMG export/import hooks compiled in
#   - test_cross_agg binary built with JACOBI_L1_2LEVEL_CONFIG (Run 4)
#
# Run on Perlmutter (interactive GPU node):
#   salloc -N 1 -C gpu -q interactive -t 01:00:00 -A m1516_g
#   bash ~/amgx-sa/scripts/run_jacobi_l1_test.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — adjust these paths for your Perlmutter environment
# ---------------------------------------------------------------------------
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$HOME/amgx-sa/build_perlmutter}"
PETSC_DIR="${PETSC_DIR:-$HOME/Codes/petsc}"
PETSC_EX2="${PETSC_EX2:-$PETSC_DIR/src/ksp/ksp/tutorials/ex2}"
AGG_DIR="${AGG_DIR:-/tmp/jacobi_l1_$$}"

# Perlmutter CUDA/math library paths (adjust if SDK version changes)
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$MATH_LIB:${LD_LIBRARY_PATH:-}"

MATRIX_FILE="$AMGX_BUILD_DIR/poisson2d.mtx"
GAMG_AGG_FILE="$AMGX_BUILD_DIR/gamg_aggs.txt"
AMGX_AGG_FILE="$AGG_DIR/amgx_aggs.txt"

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

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix file not found at $MATRIX_FILE"
    echo "Generate it with: ./examples/generate_poisson -p 5 100 100 -o $MATRIX_FILE"
    exit 1
fi

if [ ! -f "$PETSC_EX2" ]; then
    echo "WARNING: PETSc ex2 not found at $PETSC_EX2"
    echo "Test A (PETSc baseline) will be skipped."
    SKIP_PETSC=1
else
    SKIP_PETSC=0
fi

mkdir -p "$AGG_DIR"
echo "=== §7.9 Jacobi-L1 Richardson Smoother Experiment ==="
echo "Matrix: $MATRIX_FILE"
echo "Log files: $AGG_DIR"
echo ""
echo "Smoother: undamped Richardson + L1-Jacobi (d_i = sum_j |A_ij|)"
echo "Hierarchy: 2 levels (fine + coarse LU), 1 pre + 1 post sweep"
echo "Tolerance: 1e-8 relative"
echo ""

# ---------------------------------------------------------------------------
# Test A: PETSc GAMG + GAMG aggregates + Jacobi-L1 Richardson (PETSc baseline)
#         Also exports GAMG aggregates for Test B.
# ---------------------------------------------------------------------------
echo "=== Test A: PETSc GAMG + GAMG aggs + Jacobi-L1 Richardson (PETSc baseline) ==="
if [ "$SKIP_PETSC" -eq 1 ]; then
    echo "SKIPPED (PETSc ex2 not found at $PETSC_EX2)"
else
    PETSC_EXPORT_AGGREGATES="$GAMG_AGG_FILE" \
    "$PETSC_EX2" \
        -m 100 -n 100 \
        -ksp_type richardson \
        -ksp_norm_type unpreconditioned \
        -pc_type gamg \
        -pc_gamg_type agg \
        -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_mat_coarsen_type misk \
        -pc_gamg_mat_coarsen_misk_distance 2 \
        -pc_gamg_coarse_eq_limit 2000 \
        -mg_levels_ksp_type richardson \
        -mg_levels_ksp_max_it 1 \
        -mg_levels_ksp_norm_type none \
        -mg_levels_pc_type jacobi \
        -mg_levels_pc_jacobi_type rowl1 \
        -mg_levels_pc_jacobi_rowl1_scale 1 \
        -mg_levels_pc_jacobi_fixdiagonal \
        -mg_coarse_pc_type lu \
        -ksp_monitor \
        -ksp_converged_reason \
        -ksp_rtol 1e-8 \
        -ksp_view \
        -pc_gamg_verbose 1 \
        2>&1 | tee "$AGG_DIR/test_A_petsc_jacobi_l1.log"
    echo "GAMG aggregates exported to: $GAMG_AGG_FILE"
fi

# ---------------------------------------------------------------------------
# Test C: AMGx SA + AMGx MIS-2 aggregates + Jacobi-L1 Richardson
#         (AMGx baseline — Run 4 in test_cross_agg)
#         Also exports AMGx aggregates.
# ---------------------------------------------------------------------------
echo ""
echo "=== Test C: AMGx SA + AMGx MIS-2 aggs + Jacobi-L1 Richardson ==="
echo "    (Run 4 in test_cross_agg — AMGx baseline)"
cd "$AMGX_BUILD_DIR"
AMGX_EXPORT_AGGREGATES="$AMGX_AGG_FILE" \
"$AMGX_CROSS" "$MATRIX_FILE" \
    2>&1 | tee "$AGG_DIR/test_C_amgx_amgx_aggs.log"
echo "AMGx aggregates exported to: $AMGX_AGG_FILE"

# ---------------------------------------------------------------------------
# Test B: AMGx SA + GAMG aggregates + Jacobi-L1 Richardson
#         (Run 4 in test_cross_agg with GAMG aggs imported)
# ---------------------------------------------------------------------------
echo ""
echo "=== Test B: AMGx SA + GAMG aggs + Jacobi-L1 Richardson ==="
echo "    (Run 4 in test_cross_agg — GAMG aggs imported)"
if [ "$SKIP_PETSC" -eq 1 ] || [ ! -f "$GAMG_AGG_FILE" ]; then
    echo "SKIPPED (GAMG aggregate file not available: $GAMG_AGG_FILE)"
    echo "  Run Test A first (requires PETSc with GAMG export hook)"
else
    cd "$AMGX_BUILD_DIR"
    AMGX_IMPORT_AGGREGATES="$GAMG_AGG_FILE" \
    "$AMGX_CROSS" "$MATRIX_FILE" \
        2>&1 | tee "$AGG_DIR/test_B_amgx_gamg_aggs.log"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Summary: §7.9 Jacobi-L1 Richardson Results ==="
echo "Log files in: $AGG_DIR"
echo ""
echo "Iteration counts (Run 4 = Jacobi-L1 Richardson):"
echo ""

# PETSc Test A
if [ -f "$AGG_DIR/test_A_petsc_jacobi_l1.log" ]; then
    iters_A=$(grep -oP 'Linear solve converged.*iterations \K[0-9]+' \
              "$AGG_DIR/test_A_petsc_jacobi_l1.log" 2>/dev/null | tail -1 || \
              grep -oP 'KSP Object.*\K[0-9]+(?= iterations)' \
              "$AGG_DIR/test_A_petsc_jacobi_l1.log" 2>/dev/null | tail -1 || \
              echo "?")
    echo "  Test A (PETSc GAMG + GAMG aggs + Jacobi-L1): $iters_A iters"
else
    echo "  Test A (PETSc GAMG + GAMG aggs + Jacobi-L1): SKIPPED"
fi

# AMGx Test B (GAMG aggs, Run 4)
if [ -f "$AGG_DIR/test_B_amgx_gamg_aggs.log" ]; then
    # Run 4 label contains "Jacobi-L1"
    iters_B=$(grep -A5 "Jacobi-L1" "$AGG_DIR/test_B_amgx_gamg_aggs.log" 2>/dev/null | \
              grep -oP 'Iterations:\s*\K[0-9]+' | tail -1 || echo "?")
    echo "  Test B (AMGx SA  + GAMG aggs + Jacobi-L1): $iters_B iters"
else
    echo "  Test B (AMGx SA  + GAMG aggs + Jacobi-L1): SKIPPED"
fi

# AMGx Test C (AMGx aggs, Run 4)
if [ -f "$AGG_DIR/test_C_amgx_amgx_aggs.log" ]; then
    iters_C=$(grep -A5 "Jacobi-L1" "$AGG_DIR/test_C_amgx_amgx_aggs.log" 2>/dev/null | \
              grep -oP 'Iterations:\s*\K[0-9]+' | tail -1 || echo "?")
    echo "  Test C (AMGx SA  + AMGx aggs + Jacobi-L1): $iters_C iters"
else
    echo "  Test C (AMGx SA  + AMGx aggs + Jacobi-L1): SKIPPED"
fi

echo ""
echo "SA prolongator omega (from [SA-VIEW] lines in AMGx logs):"
for log in "$AGG_DIR"/test_B_amgx_gamg_aggs.log "$AGG_DIR"/test_C_amgx_amgx_aggs.log; do
    [ -f "$log" ] || continue
    label=$(basename "$log" .log)
    grep "\[SA-VIEW\]" "$log" 2>/dev/null | head -4 || true
done

echo ""
echo "Interpretation:"
echo "  B ≈ A: SA prolongator construction is equivalent to GAMG's"
echo "  B > A: gap is in prolongator construction (P_tent QR, P_smooth omega, Galerkin RAP)"
echo "  C > B: aggregate quality also contributes (AMGx MIS-2 vs GAMG MIS-2)"
echo ""
echo "Note: [SA-VIEW] lines show rho(D^{-1}A) and omega=(4/3)/rho for SA prolongator smoothing."
echo "      PETSc GAMG uses omega=1.4/rho (same formula, different constant)."
echo "      Check -ksp_view output in Test A log for PETSc's omega value."
