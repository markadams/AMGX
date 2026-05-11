#!/bin/bash
# =============================================================================
# run_mg_diag_compare.sh
#
# MG-DIAG Comparison: PETSc GAMG vs AMGx SA
#
# Runs two tests with matching [MG-DIAG] instrumentation to compare
# per-stage norms in the V-cycle and find where the coarse grid correction
# diverges between PETSc and AMGx.
#
# Test 1: PETSc GAMG ex2 — Jacobi-L1 Richardson, 2-level, MIS-k=2
# Test 2: AMGx test_mg_diag — Jacobi-L1 Richardson, 2-level, GAMG aggs imported
#
# Both print [MG-DIAG] lines at 7 points in the V-cycle:
#   ENTRY, AFTER-PRESMOOTH, AFTER-RESIDUAL, AFTER-RESTRICT,
#   AFTER-COARSE-SOLVE, AFTER-PROLONGATE, AFTER-POSTSMOOTH
#
# Run on Perlmutter (interactive GPU node):
#   salloc -N 1 -C gpu -q interactive -t 00:30:00 -A m1516_g
#   bash ~/amgx-sa/scripts/run_mg_diag_compare.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AMGX_DIR="${AMGX_DIR:-$HOME/amgx-sa}"
AMGX_BUILD_DIR="${AMGX_BUILD_DIR:-$AMGX_DIR/build_perlmutter}"
PETSC_DIR="${PETSC_DIR:-$HOME/petsc}"
PETSC_ARCH="${PETSC_ARCH:-arch-perlmutter-dbg-gcc-cuda}"
PETSC_EX2="${PETSC_EX2:-$PETSC_DIR/src/ksp/ksp/tutorials/ex2}"

# Output directory
OUT_DIR="${OUT_DIR:-$HOME/mg_diag_results}"
mkdir -p "$OUT_DIR"

# Perlmutter CUDA/math library paths
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$MATH_LIB:${LD_LIBRARY_PATH:-}"

# Files
MATRIX_FILE="$AMGX_BUILD_DIR/poisson2d.mtx"
GAMG_AGG_FILE="$AMGX_BUILD_DIR/gamg_aggs.txt"
AMGX_TEST="$AMGX_BUILD_DIR/test_mg_diag"

echo "============================================================"
echo "  MG-DIAG Comparison: PETSc GAMG vs AMGx SA"
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
    echo "  Build it: cd $PETSC_DIR/src/ksp/ksp/tutorials && make ex2"
    SKIP_PETSC=1
else
    SKIP_PETSC=0
fi

if [ ! -f "$AMGX_TEST" ]; then
    echo "ERROR: test_mg_diag not found at $AMGX_TEST"
    echo "  Build it from $AMGX_BUILD_DIR:"
    echo "    /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \\"
    echo "      ../examples/test_mg_diag.c -o test_mg_diag \\"
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
# Test 1: PETSc GAMG + Jacobi-L1 Richardson + MG-DIAG
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Test 1: PETSc GAMG + Jacobi-L1 Richardson"
echo "============================================================"

PETSC_LOG="$OUT_DIR/test1_petsc_mg_diag.log"

if [ "$SKIP_PETSC" -eq 1 ]; then
    echo "SKIPPED (PETSc ex2 not found)"
else
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
        2>&1 | tee "$PETSC_LOG"

    echo ""
    echo "PETSc log saved to: $PETSC_LOG"
fi

# ---------------------------------------------------------------------------
# Test 2: AMGx + GAMG aggregates + Jacobi-L1 Richardson + MG-DIAG
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Test 2: AMGx + GAMG aggs + Jacobi-L1 Richardson"
echo "============================================================"

AMGX_LOG="$OUT_DIR/test2_amgx_mg_diag.log"

cd "$AMGX_BUILD_DIR"
AMGX_IMPORT_AGGREGATES="$GAMG_AGG_FILE" \
"$AMGX_TEST" "$MATRIX_FILE" \
    2>&1 | tee "$AMGX_LOG"

echo ""
echo "AMGx log saved to: $AMGX_LOG"

# ---------------------------------------------------------------------------
# Extract and compare MG-DIAG lines
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  MG-DIAG Comparison (first 5 iterations)"
echo "============================================================"

PETSC_DIAG="$OUT_DIR/petsc_mg_diag.txt"
AMGX_DIAG="$OUT_DIR/amgx_mg_diag.txt"

if [ -f "$PETSC_LOG" ]; then
    grep '\[MG-DIAG\]' "$PETSC_LOG" > "$PETSC_DIAG" 2>/dev/null || true
    echo ""
    echo "--- PETSc [MG-DIAG] ---"
    cat "$PETSC_DIAG"
fi

if [ -f "$AMGX_LOG" ]; then
    grep '\[MG-DIAG\]' "$AMGX_LOG" > "$AMGX_DIAG" 2>/dev/null || true
    echo ""
    echo "--- AMGx [MG-DIAG] ---"
    cat "$AMGX_DIAG"
fi

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
echo ""
echo "Log files:"
echo "  PETSc: $PETSC_LOG"
echo "  AMGx:  $AMGX_LOG"
echo ""
echo "MG-DIAG extracts:"
echo "  PETSc: $PETSC_DIAG"
echo "  AMGx:  $AMGX_DIAG"
echo ""
echo "NOTE: PETSc level numbering is OPPOSITE to AMGx:"
echo "  PETSc: level 0 = coarsest, level 1 = finest (for 2-level)"
echo "  AMGx:  level 0 = finest,   level 1 = coarsest"
echo "  Compare by stage name (ENTRY, AFTER-PRESMOOTH, etc.), not level number."
echo ""
echo "Look for the first stage where norms diverge significantly."
echo "============================================================"
