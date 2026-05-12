// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// test_multilevel_cheby.c
//
// Test driver: PCG outer solver with SA-AMG preconditioner using
// Chebyshev(1)+Jacobi smoother. Multi-level V(1,1) cycle.
//
// Configuration:
//   - Outer: PCG, tolerance 1e-8, RELATIVE_INI convergence
//   - Preconditioner: SA-AMG V(1,1) cycle
//   - Smoother: CHEBYSHEV order=1, BLOCK_JACOBI relaxation_factor=1.0
//   - chebyshev_lambda_estimate_mode=4, chebyshev_lmin_denom=11
//   - max_levels=20, min_coarse_rows=10 (deep hierarchy)
//   - Coarse solver: DENSE_LU_SOLVER
//   - Selector: MIS-2
//
// Build on Perlmutter (from build_perlmutter/):
//   CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
//   MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
//   /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \
//     ../examples/test_multilevel_cheby.c -o test_multilevel_cheby \
//     ./libamgxsh.so -lrt -ldl \
//     $CUDA_LIB/libcudart_static.a \
//     $MATH_LIB/libcublas.so $MATH_LIB/libcusolver.so $MATH_LIB/libcusparse.so \
//     $CUDA_LIB/libculibos.a $CUDA_LIB/libnvJitLink.so $MATH_LIB/libcublasLt.so \
//     -lm -lpthread -ldl
//
// Run:
//   ./test_multilevel_cheby poisson2d_400.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "amgx_c.h"

static void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}

// -----------------------------------------------------------------------
// PCG outer solver + SA-AMG preconditioner with Chebyshev(1)+Jacobi smoother.
// Multi-level: max_levels=20, min_coarse_rows=10
// -----------------------------------------------------------------------
static const char *PCG_CHEBY_MULTILEVEL_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"solver\": \"PCG\","
    "    \"preconditioner\": {"
    "      \"algorithm\": \"AGGREGATION\","
    "      \"solver\": \"AMG\","
    "      \"smoother\": {"
    "        \"solver\": \"CHEBYSHEV\","
    "        \"chebyshev_polynomial_order\": 1,"
    "        \"chebyshev_lambda_estimate_mode\": 4,"
    "        \"chebyshev_lmin_denom\": 11.0,"
    "        \"preconditioner\": {"
    "          \"solver\": \"BLOCK_JACOBI\","
    "          \"relaxation_factor\": 1.0,"
    "          \"max_iters\": 1,"
    "          \"monitor_residual\": 0,"
    "          \"print_solve_stats\": 0"
    "        },"
    "        \"max_iters\": 1,"
    "        \"monitor_residual\": 0,"
    "        \"print_solve_stats\": 0"
    "      },"
    "      \"presweeps\": 1,"
    "      \"postsweeps\": 1,"
    "      \"selector\": \"MIS\","
    "      \"mis_k\": 2,"
    "      \"aggressive_levels\": 1,"
    "      \"max_aggregate_size\": 12,"
    "      \"refine_threshold\": 0.0,"
    "      \"merge_singletons\": 1,"
    "      \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "      \"max_iters\": 1,"
    "      \"min_coarse_rows\": 10,"
    "      \"max_levels\": 20,"
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
        fprintf(stderr, "PCG + SA-AMG with Chebyshev(1)+Jacobi smoother, multi-level V(1,1) cycle.\n");
        fprintf(stderr, "max_levels=20, min_coarse_rows=10\n");
        return 1;
    }

    const char *matrix_file = argv[1];

    printf("=== AMGx Multi-Level PCG + Chebyshev(1)+Jacobi SA-AMG Test ===\n");
    printf("Matrix file: %s\n", matrix_file);
    printf("Config: PCG outer, SA-AMG preconditioner\n");
    printf("  Smoother: Chebyshev(1)+Jacobi, V(1,1) cycle\n");
    printf("  max_levels=20, min_coarse_rows=10\n");
    printf("  chebyshev_lambda_estimate_mode=4, chebyshev_lmin_denom=11\n");
    printf("  Coarse solver: DENSE_LU_SOLVER\n");
    printf("  Selector: MIS-2\n");
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

    AMGX_config_create(&cfg, PCG_CHEBY_MULTILEVEL_CONFIG);
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

    // Setup (builds hierarchy)
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
