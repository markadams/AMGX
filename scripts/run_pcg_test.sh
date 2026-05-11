#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -t 00:15:00
#SBATCH -J cross_agg_pcg
#SBATCH -o /global/homes/m/madams/cross_agg_pcg_%j.out
#SBATCH -e /global/homes/m/madams/cross_agg_pcg_%j.err
#SBATCH -A m1516_g

# Run from ~/amgx-sa/build_perlmutter/ on Perlmutter.
# Submits all 3 AMGx config variants (Run 1, 2, 3) for both aggregate sources.
#
# Usage (from login node):
#   sbatch ~/amgx-sa/scripts/run_pcg_test.sh

AMGX_BUILD_DIR=$HOME/amgx-sa/build_perlmutter
AGG_DIR=$AMGX_BUILD_DIR

CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/23.1/cuda/12.0/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
export LD_LIBRARY_PATH="$AMGX_BUILD_DIR:$CUDA_LIB:$MATH_LIB:${LD_LIBRARY_PATH:-}"

MATRIX=$AMGX_BUILD_DIR/poisson2d.mtx

echo "=== Step 7.8: PCG outer solver experiment ==="
echo "Matrix: $MATRIX"
echo "Date: $(date)"
echo ""

echo "=== Tests with AMGx MIS-2 aggregates (export) ==="
AMGX_EXPORT_AGGREGATES=$AGG_DIR/amgx_aggs.txt \
  srun -n1 --gpus=1 $AMGX_BUILD_DIR/test_cross_agg $MATRIX 2>&1

echo ""
echo "=== Tests with GAMG aggregates (import) ==="
AMGX_IMPORT_AGGREGATES=$AGG_DIR/gamg_aggs.txt \
  srun -n1 --gpus=1 $AMGX_BUILD_DIR/test_cross_agg $MATRIX 2>&1
