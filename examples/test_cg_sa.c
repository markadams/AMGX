// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// test_cg_sa.c
//
// Test driver: CG outer solver + SA-AMG preconditioner (Jacobi-L1 smoother).
// Compares against PETSc:
//   -ksp_type cg -ksp_norm_type unpreconditioned
//   -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1
//   -pc_gamg_mat_coarsen_type misk -pc_gamg_mat_coarsen_misk_distance 2
//   -pc_gamg_coarse_eq_limit 2000
//   -mg_levels_ksp_type richardson -mg_levels_ksp_max_it 1
//   -mg_levels_pc_type jacobi -mg_levels_pc_jacobi_type rowl1
//   -mg_levels_pc_jacobi_rowl1_scale 1 -mg_levels_pc_jacobi_fixdiagonal
//   -mg_coarse_pc_type lu
//   -ksp_rtol 1e-8
//
// Build on Perlmutter (from build_perlmutter/):
//   CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
//   MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
//   /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \
//     ../examples/test_cg_sa.c -o test_cg_sa \
//     ./libamgxsh.so -lrt -ldl \
//     $CUDA_LIB/libcudart_static.a \
//     $MATH_LIB/libcublas.so $MATH_LIB/libcusolver.so $MATH_LIB/libcusparse.so \
//     $CUDA_LIB/libculibos.a $CUDA_LIB/libnvJitLink.so $MATH_LIB/libcublasLt.so \
//     -lm -lpthread -ldl
//
// Run:
//   AMGX_IMPORT_AGGREGATES=gamg_aggs.txt ./test_cg_sa poisson2d.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amgx_c.h"

static void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}

// -----------------------------------------------------------------------
// PCG + SA-AMG preconditioner.
//
// Outer solver: PCG (Conjugate Gradient)
//   - convergence: RELATIVE_INI (||r_k||/||r_0|| < tol), matches PETSc rtol
//   - norm: L2 (unpreconditioned residual norm, matches -ksp_norm_type unpreconditioned)
//
// Preconditioner: 1 V-cycle of SA-AMG
//   - smoother: Jacobi-L1, relaxation_factor=1.0, 1 sweep
//   - presweeps=1, postsweeps=1
//   - coarse solver: Dense LU
//   - 2 levels max
// -----------------------------------------------------------------------
static const char *PCG_SA_AMG_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"solver\": \"PCG\","
    "    \"preconditioner\": {"
    "      \"algorithm\": \"AGGREGATION\","
    "      \"solver\": \"AMG\","
    "      \"smoother\": {"
    "        \"solver\": \"JACOBI_L1\","
    "        \"relaxation_factor\": 1.0,"
    "        \"max_iters\": 1,"
    "        \"monitor_residual\": 0,"
    "        \"print_solve_stats\": 0"
    "      },"
    "      \"presweeps\": 1,"
    "      \"postsweeps\": 1,"
    "      \"selector\": \"MIS\","
    "      \"mis_k\": 2,"
    "      \"aggressive_levels\": 1,"
    "      \"merge_singletons\": 1,"
    "      \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "      \"max_iters\": 1,"
    "      \"min_coarse_rows\": 2,"
    "      \"max_levels\": 2,"
    "      \"cycle\": \"V\","
    "      \"print_solve_stats\": 0,"
    "      \"print_grid_stats\": 1,"
    "      \"monitor_residual\": 0,"
    "      \"obtain_timings\": 0"
    "    },"
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 0,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "CG+SA-AMG test: PCG outer solver with Jacobi-L1 SA-AMG preconditioner.\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Environment variables:\n");
        fprintf(stderr, "  AMGX_IMPORT_AGGREGATES=<file>  import aggregates from GAMG\n");
        fprintf(stderr, "  AMGX_EXPORT_AGGREGATES=<file>  export AMGx aggregates\n");
        return 1;
    }

    const char *matrix_file = argv[1];
    const char *import_file = getenv("AMGX_IMPORT_AGGREGATES");
    const char *export_file = getenv("AMGX_EXPORT_AGGREGATES");

    printf("=== AMGx CG+SA-AMG Test ===\n");
    printf("Matrix file: %s\n", matrix_file);
    printf("Config: PCG outer solver + Jacobi-L1 SA-AMG preconditioner\n");
    if (import_file)
        printf("Importing aggregates from: %s\n", import_file);
    else
        printf("Using AMGx MIS-2 aggregates (export to: %s)\n",
               export_file ? export_file : "(not exporting)");
    printf("\n");

    AMGX_initialize();
    AMGX_initialize_plugins();
    AMGX_register_print_callback(&print_callback);
    AMGX_install_signal_handler();

    AMGX_config_handle    cfg;
    AMGX_resources_handle res;
    AMGX_matrix_handle    mtx;
    AMGX_vector_handle    b, x;
    AMGX_solver_handle    slv;
    AMGX_SOLVE_STATUS     status;

    AMGX_config_create(&cfg, PCG_SA_AMG_CONFIG);
    AMGX_resources_create_simple(&res, cfg);
    AMGX_matrix_create(&mtx, res, AMGX_mode_dDDI);
    AMGX_vector_create(&b,   res, AMGX_mode_dDDI);
    AMGX_vector_create(&x,   res, AMGX_mode_dDDI);
    AMGX_solver_create(&slv, res, AMGX_mode_dDDI, cfg);

    // Read matrix
    AMGX_read_system(mtx, b, x, matrix_file);

    // Get matrix size
    int n, bsize_x, bsize_y;
    AMGX_matrix_get_size(mtx, &n, &bsize_x, &bsize_y);
    printf("Matrix: %d rows, block_size=%dx%d\n", n, bsize_x, bsize_y);

    // Zero initial guess
    AMGX_vector_set_zero(x, n, bsize_x);

    // Set near-null space (constant vector) BEFORE setup
    {
        double *null_data = (double *)malloc(n * sizeof(double));
        for (int i = 0; i < n; i++)
            null_data[i] = 1.0;
        AMGX_solver_set_near_null_space(slv, 1, n, null_data);
        free(null_data);
    }

    // Setup (builds hierarchy, imports/exports aggregates)
    printf("\n--- Setup ---\n");
    AMGX_solver_setup(slv, mtx);

    // Solve
    printf("\n--- Solve ---\n");
    AMGX_solver_solve(slv, b, x);
    AMGX_solver_get_status(slv, &status);

    printf("\n--- Result ---\n");
    printf("Solve status: %s\n",
           status == AMGX_SOLVE_SUCCESS ? "CONVERGED" :
           status == AMGX_SOLVE_DIVERGED ? "DIVERGED" : "NOT_CONVERGED");

    AMGX_solver_destroy(slv);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(mtx);
    AMGX_resources_destroy(res);
    AMGX_config_destroy(cfg);

    AMGX_finalize_plugins();
    AMGX_finalize();
    return (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}
