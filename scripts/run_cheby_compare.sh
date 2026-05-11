#!/bin/bash
# =============================================================================
# run_cheby_compare.sh
#
# Chebyshev(1)+Jacobi SA-AMG Comparison: PETSc GAMG vs AMGx SA
#
# Changes only the smoother from Jacobi-L1 to Chebyshev(1)+Jacobi, using
# Richardson outer solver (standalone AMG) with imported GAMG aggregates.
# V(1,1) cycle.
#
# Eigenvalue interval matching:
#   PETSc GAMG: lmax = 1.1 * rho(D^{-1}A),  lmin = 0.1 * lmax
#   AMGx:       chebyshev_lambda_estimate_mode=4 → lmax = 1.1 * rho (SA rho reused)
#               chebyshev_lmin_denom=10          → lmin = lmax / 10 = 0.1 * lmax
#
# Test 1: PETSc ex2 — Richardson outer, GAMG, Chebyshev(1)+Jacobi smoother
# Test 2: AMGx test_cheby_sa — standalone SA-AMG, Chebyshev(1)+Jacobi, GAMG aggs
#
# Run on Perlmutter (interactive GPU node):
#   salloc -N 1 -C gpu -q interactive -t 00:30:00 -A m1516_g
#   bash ~/amgx-sa/scripts/run_cheby_compare.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AMGX_DIR="${AMGX_DIR:-$HOME/amgx-sa}"
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$AMGX_DIR/build_perlmutter}"
PETSC_DIR="${PETSC_DIR:-$HOME/petsc}"
PETSC_EX2="${PETSC_EX2:-$PETSC_DIR/src/ksp/ksp/tutorials/ex2}"

OUT_DIR="${OUT_DIR:-$HOME/cheby_compare_results}"
mkdir -p "$OUT_DIR"

CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$MATH_LIB:${LD_LIBRARY_PATH:-}"

MATRIX_FILE="$AMGX_BUILD_DIR/poisson2d.mtx"
GAMG_AGG_FILE="$AMGX_BUILD_DIR/gamg_aggs.txt"
AMGX_TEST="$AMGX_BUILD_DIR/test_cheby_sa"

echo "============================================================"
echo "  Chebyshev(1)+Jacobi SA-AMG Comparison: PETSc GAMG vs AMGx SA"
echo "============================================================"
echo ""
echo "Output directory: $OUT_DIR"
echo "Matrix: $MATRIX_FILE"
echo "GAMG aggregates: $GAMG_AGG_FILE"
echo ""

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------
ERRORS=0

if [ ! -f "$PETSC_EX2" ]; then
    echo "WARNING: PETSc ex2 not found at $PETSC_EX2"
    echo "  Test 1 (PETSc) will be skipped."
    SKIP_PETSC=1
else
    SKIP_PETSC=0
fi

if [ ! -f "$AMGX_TEST" ]; then
    echo "ERROR: test_cheby_sa not found at $AMGX_TEST"
    echo "  Build it from $AMGX_BUILD_DIR:"
    echo "    /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \\"
    echo "      ../examples/test_cheby_sa.c -o test_cheby_sa \\"
    echo "      ./libamgxsh.so -lrt -ldl \\"
    echo "      $CUDA_LIB/libcudart_static.a \\"
    echo "      $MATH_LIB/libcublas.so $MATH_LIB/libcusolver.so $MATH_LIB/libcusparse.so \\"
    echo "      $CUDA_LIB/libculibos.a $CUDA_LIB/libnvJitLink.so $MATH_LIB/libcublasLt.so \\"
    echo "      -lm -lpthread -ldl"
    ERRORS=1
fi

if [ ! -f "$MATRIX_FILE" ]; then
    echo "ERROR: Matrix file not found at $MATRIX_FILE"
    ERRORS=1
fi

if [ ! -f "$GAMG_AGG_FILE" ]; then
    echo "ERROR: GAMG aggregate file not found at $GAMG_AGG_FILE"
    ERRORS=1
fi

if [ "$ERRORS" -eq 1 ]; then
    echo ""
    echo "Fix the errors above and re-run."
    exit 1
fi

# ---------------------------------------------------------------------------
# Test 1: PETSc CG + GAMG preconditioner + Chebyshev(1)+Jacobi smoother
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Test 1: PETSc Richardson + GAMG + Chebyshev(1)+Jacobi smoother"
echo "============================================================"

PETSC_LOG="$OUT_DIR/test1_petsc_cheby.log"

if [ "$SKIP_PETSC" -eq 1 ]; then
    echo "SKIPPED (PETSc ex2 not found)"
else
    # Export GAMG aggregates so AMGx can import them
    PETSC_EXPORT_AGGREGATES="$GAMG_AGG_FILE" \
    "$PETSC_EX2" \
        -m 100 -n 100 \
        -ksp_type richardson \
        -ksp_richardson_scale 1.0 \
        -ksp_norm_type unpreconditioned \
        -pc_type gamg \
        -pc_gamg_type agg \
        -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_mat_coarsen_type misk \
        -pc_gamg_mat_coarsen_misk_distance 2 \
        -pc_gamg_coarse_eq_limit 2000 \
        -mg_levels_ksp_type chebyshev \
        -mg_levels_ksp_max_it 1 \
        -mg_levels_pc_type jacobi \
        -mg_coarse_pc_type lu \
        -ksp_monitor \
        -ksp_converged_reason \
        -ksp_rtol 1e-8 \
        2>&1 | tee "$PETSC_LOG"

    echo ""
    echo "PETSc log saved to: $PETSC_LOG"
fi

# ---------------------------------------------------------------------------
# Test 2: AMGx PCG + SA-AMG + Chebyshev(1)+Jacobi + imported GAMG aggregates
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Test 2: AMGx PCG + SA-AMG + Chebyshev(1)+Jacobi + GAMG aggs"
echo "============================================================"

AMGX_LOG="$OUT_DIR/test2_amgx_cheby.log"

cd "$AMGX_BUILD_DIR"
AMGX_IMPORT_AGGREGATES="$GAMG_AGG_FILE" \
"$AMGX_TEST" "$MATRIX_FILE" \
    2>&1 | tee "$AMGX_LOG"

echo ""
echo "AMGx log saved to: $AMGX_LOG"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo ""

PETSC_ITERS=""
AMGX_ITERS=""

if [ -f "$PETSC_LOG" ]; then
    PETSC_ITERS=$(grep "Linear solve converged" "$PETSC_LOG" | grep -oE "iterations [0-9]+" | grep -oE "[0-9]+" || echo "?")
    echo "PETSc Richardson+Chebyshev(1) iterations: $PETSC_ITERS"
fi

if [ -f "$AMGX_LOG" ]; then
    AMGX_ITERS=$(grep "Total Iterations:" "$AMGX_LOG" | grep -oE "[0-9]+" || echo "?")
    echo "AMGx AMG+Chebyshev(1) iterations: $AMGX_ITERS"
fi

echo ""
if [ "$PETSC_ITERS" = "$AMGX_ITERS" ] && [ -n "$PETSC_ITERS" ] && [ "$PETSC_ITERS" != "?" ]; then
    echo "RESULT: MATCH — both converge in $PETSC_ITERS iterations ✅"
else
    echo "RESULT: MISMATCH — PETSc=$PETSC_ITERS  AMGx=$AMGX_ITERS ❌"
fi

echo ""
echo "Log files:"
echo "  PETSc: $PETSC_LOG"
echo "  AMGx:  $AMGX_LOG"
echo "============================================================"
