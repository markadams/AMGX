#!/bin/bash
# Run this in your interactive Perlmutter shell (already on a GPU node).
# Usage: bash ~/run_cross_agg_interactive.sh 2>&1 | tee /tmp/cross_agg_results.txt

AMGX_BUILD_DIR=$HOME/amgx-sa/build_perlmutter
PETSC_DIR=$HOME/petsc
PETSC_EX2=$PETSC_DIR/src/ksp/ksp/tutorials/ex2
AGG_DIR=/tmp/cross_agg_test
mkdir -p $AGG_DIR

# Use the 100x100 Poisson already generated (10000x10000)
MATRIX_FILE=$AMGX_BUILD_DIR/poisson2d.mtx

GAMG_AGG_FILE=$AGG_DIR/gamg_aggregates.txt
AMGX_AGG_FILE=$AGG_DIR/amgx_aggregates.txt

# Correct CUDA lib path for this node's loaded modules
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
PETSC_LIB=$PETSC_DIR/arch-perlmutter-dbg-gcc-cuda/lib
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$CUDA_LIB:$MATH_LIB:$PETSC_LIB:${LD_LIBRARY_PATH:-}"

PETSC_OPTS="-m 100 -n 100 \
  -pc_type gamg -pc_gamg_type agg \
  -pc_gamg_agg_nsmooths 1 \
  -pc_gamg_mat_coarsen_type misk \
  -pc_gamg_mat_coarsen_misk_distance 2 \
  -mg_levels_ksp_type chebyshev \
  -mg_levels_ksp_max_it 1 \
  -mg_levels_pc_type jacobi \
  -mg_coarse_pc_type lu \
  -ksp_type cg \
  -ksp_rtol 1e-8 \
  -ksp_monitor_short \
  -ksp_converged_reason"

echo "Matrix: $MATRIX_FILE ($(head -2 $MATRIX_FILE | tail -1))"

echo ""
echo "=== Test A: PETSc solver + GAMG aggregates (PETSc baseline) ==="
PETSC_EXPORT_AGGREGATES=$GAMG_AGG_FILE \
  $PETSC_EX2 $PETSC_OPTS 2>&1 | grep -E 'KSP Residual|converged|CROSS-AGG'
echo "GAMG agg file: $(head -4 $GAMG_AGG_FILE 2>/dev/null || echo 'NOT CREATED')"

echo ""
echo "=== Test D: AMGx solver + AMGx aggregates (AMGx baseline) ==="
AMGX_EXPORT_AGGREGATES=$AMGX_AGG_FILE \
  $AMGX_BUILD_DIR/test_cross_agg $MATRIX_FILE 2>&1 | \
  grep -E 'iter|Iter|CROSS-AGG|converged|residual|Residual|Solve|solve'
echo "AMGx agg file: $(head -4 $AMGX_AGG_FILE 2>/dev/null || echo 'NOT CREATED')"

echo ""
echo "=== Test B: AMGx solver + GAMG aggregates ==="
AMGX_IMPORT_AGGREGATES=$GAMG_AGG_FILE \
  $AMGX_BUILD_DIR/test_cross_agg $MATRIX_FILE 2>&1 | \
  grep -E 'iter|Iter|CROSS-AGG|converged|residual|Residual|Solve|solve'

echo ""
echo "=== Test C: PETSc solver + AMGx aggregates ==="
PETSC_IMPORT_AGGREGATES=$AMGX_AGG_FILE \
  $PETSC_EX2 $PETSC_OPTS 2>&1 | grep -E 'KSP Residual|converged|CROSS-AGG'

echo ""
echo "=== Summary ==="
echo "Test A (GAMG aggs + PETSc): see KSP iterations above"
echo "Test B (GAMG aggs + AMGx):  see AMGx iterations above"
echo "Test C (AMGx aggs + PETSc): see KSP iterations above"
echo "Test D (AMGx aggs + AMGx):  see AMGx iterations above"
