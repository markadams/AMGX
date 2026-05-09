// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// test_sa_phase1.c
//
// Verification test for Phase 1: SA solve-cycle fix.
//
// Reads a matrix from a file (MatrixMarket format) and solves Ax=b twice:
//   Run 1: standard AGGREGATION (no near-null space)
//   Run 2: SA AGGREGATION (constant near-null space = all-ones)
//
// With the Phase 1 fix, Run 2 uses the smoothed prolongator P for both
// grid transfer and the Galerkin coarse-A, giving a mathematically
// consistent V-cycle.  The SA run should converge in fewer iterations.
//
// Build on Perlmutter:
//   cd ~/amgx-sa/build_perlmutter
//   /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \
//     ../examples/test_sa_phase1.c -o test_sa_phase1 \
//     ./libamgxsh.so -lrt -ldl \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib/libcudart_static.a \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/libcublas.so \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/libcusolver.so \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/libcusparse.so \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib/libculibos.a \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib/libnvJitLink.so \
//     /opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64/libcublasLt.so \
//     -lm -lpthread -ldl
//
// Run:
//   # Generate 2D Poisson matrix first:
//   ./examples/generate_poisson -p 5 100 100 -o poisson2d.mtx
//   LD_LIBRARY_PATH=.:/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64:... \
//     ./test_sa_phase1 poisson2d.mtx

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "amgx_c.h"

static void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}

// -----------------------------------------------------------------------
// Config 1: damped Jacobi (omega=2/3), V(2,2) cycle, SIZE_4 selector,
// DENSE_LU_SOLVER on coarsest grid for exact coarse solve.
// omega=2/3 is the classical optimal damping for Jacobi on the Laplacian
// (Trottenberg et al., Section 2.2).
// -----------------------------------------------------------------------
static const char *JACOBI_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"BLOCK_JACOBI\","
    "      \"relaxation_factor\": 0.6667,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 2,"
    "    \"postsweeps\": 2,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// SA variant: same as JACOBI_CONFIG but coarse_solver=JACOBI_L1 (20 sweeps)
// because P^T A P from the smoothed prolongator can be near-singular.
static const char *JACOBI_SA_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"BLOCK_JACOBI\","
    "      \"relaxation_factor\": 0.6667,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 2,"
    "    \"postsweeps\": 2,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Config 2: symmetric Gauss-Seidel (SOR omega=1), V(1,1) cycle,
// SIZE_4 selector, DENSE_LU_SOLVER on coarsest grid.
// One sym-GS sweep = one forward + one backward pass (already symmetric).
// Sym-GS satisfies the smoothing property without damping and is the
// standard smoother for classical AMG (Trottenberg et al., Ch. 4).
// -----------------------------------------------------------------------
static const char *SGS_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": \"MULTICOLOR_GS\","
    "    \"symmetric_GS\": 1,"
    "    \"matrix_coloring_scheme\": \"MIN_MAX\","
    "    \"presweeps\": 1,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// SA variant: same as SGS_CONFIG but coarse_solver=JACOBI_L1 (20 sweeps).
static const char *SGS_SA_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": \"MULTICOLOR_GS\","
    "    \"symmetric_GS\": 1,"
    "    \"matrix_coloring_scheme\": \"MIN_MAX\","
    "    \"presweeps\": 1,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Config 3: Chebyshev smoother (order 4) with JACOBI_L1 preconditioner.
// lambda_estimate_mode=3: power iteration for lambda_max.
// V(1,1) cycle (one Chebyshev application = 4 matrix-vector products).
// SIZE_4 selector, DENSE_LU_SOLVER on coarsest grid.
// -----------------------------------------------------------------------
static const char *CHEBY_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"CHEBYSHEV\","
    "      \"preconditioner\": {"
    "        \"solver\": \"JACOBI_L1\","
    "        \"max_iters\": 1"
    "      },"
    "      \"max_iters\": 1,"
    "      \"chebyshev_polynomial_order\": 4,"
    "      \"chebyshev_lambda_estimate_mode\": 3,"
    "      \"cheby_max_lambda\": 2.0,"
    "      \"cheby_min_lambda\": 0.222,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 0,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// SA variant: same as CHEBY_CONFIG but coarse_solver=JACOBI_L1 (20 sweeps).
static const char *CHEBY_SA_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"CHEBYSHEV\","
    "      \"preconditioner\": {"
    "        \"solver\": \"JACOBI_L1\","
    "        \"max_iters\": 1"
    "      },"
    "      \"max_iters\": 1,"
    "      \"chebyshev_polynomial_order\": 4,"
    "      \"chebyshev_lambda_estimate_mode\": 3,"
    "      \"cheby_max_lambda\": 2.0,"
    "      \"cheby_min_lambda\": 0.222,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 0,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"SIZE_8\","
    "    \"coarse_solver\": \"JACOBI_L1\","
    "    \"coarsest_sweeps\": 20,"
    "    \"max_iters\": 20,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 100,"
    "    \"max_levels\": 20,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Run one solve using AMGX_read_system to load the matrix from file.
// If null_dim > 0, call set_near_null_space before setup.
// -----------------------------------------------------------------------
static void run_solve(const char *label,
                      const char *matrix_file,
                      const char *config_str,
                      int null_dim)
{
    printf("\n=== %s ===\n", label);

    AMGX_config_handle    cfg;
    AMGX_resources_handle res;
    AMGX_matrix_handle    mtx;
    AMGX_vector_handle    b, x;
    AMGX_solver_handle    slv;

    AMGX_config_create(&cfg, config_str);
    AMGX_resources_create_simple(&res, cfg);
    AMGX_matrix_create(&mtx, res, AMGX_mode_dDDI);
    AMGX_vector_create(&b,   res, AMGX_mode_dDDI);
    AMGX_vector_create(&x,   res, AMGX_mode_dDDI);
    AMGX_solver_create(&slv, res, AMGX_mode_dDDI, cfg);

    // Read matrix (and rhs/solution if present in file)
    AMGX_read_system(mtx, b, x, matrix_file);

    // Get matrix size for null space
    int n, bsize_x, bsize_y;
    AMGX_matrix_get_size(mtx, &n, &bsize_x, &bsize_y);

    // Zero out initial guess
    AMGX_vector_set_zero(x, n, bsize_x);

    // SA: set near-null space BEFORE setup (required by API)
    if (null_dim > 0)
    {
        double *null_data = (double *)malloc(n * null_dim * sizeof(double));
        for (int i = 0; i < n * null_dim; i++)
            null_data[i] = 1.0;
        AMGX_solver_set_near_null_space(slv, null_dim, n, null_data);
        free(null_data);
    }

    // Setup solver (runs hierarchy construction)
    AMGX_solver_setup(slv, mtx);

    AMGX_solver_solve(slv, b, x);

    AMGX_solver_destroy(slv);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(mtx);
    AMGX_resources_destroy(res);
    AMGX_config_destroy(cfg);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        fprintf(stderr, "  Generate a 2D Poisson matrix first:\n");
        fprintf(stderr, "  ./examples/generate_poisson -p 5 100 100 -o poisson2d.mtx\n");
        return 1;
    }

    const char *matrix_file = argv[1];
    printf("Matrix file: %s\n", matrix_file);

    AMGX_initialize();
    AMGX_initialize_plugins();
    AMGX_register_print_callback(&print_callback);
    AMGX_install_signal_handler();

    // Damped Jacobi (omega=2/3): standard vs SA
    // SA uses JACOBI_L1 coarse solver because P^T A P can be near-singular.
    run_solve("Standard AGGREGATION, Jacobi omega=2/3, DENSE_LU coarse",  matrix_file, JACOBI_CONFIG,    0);
    run_solve("SA AGGREGATION,       Jacobi omega=2/3, JACOBI_L1 coarse", matrix_file, JACOBI_SA_CONFIG, 1);

    // Symmetric Gauss-Seidel (SOR omega=1): standard vs SA
    run_solve("Standard AGGREGATION, sym-GS, DENSE_LU coarse",  matrix_file, SGS_CONFIG,    0);
    run_solve("SA AGGREGATION,       sym-GS, JACOBI_L1 coarse", matrix_file, SGS_SA_CONFIG, 1);

    // Chebyshev order-4 + JACOBI_L1 precond: standard vs SA
    run_solve("Standard AGGREGATION, Chebyshev-4+L1, DENSE_LU coarse",  matrix_file, CHEBY_CONFIG,    0);
    run_solve("SA AGGREGATION,       Chebyshev-4+L1, JACOBI_L1 coarse", matrix_file, CHEBY_SA_CONFIG, 1);

    AMGX_finalize_plugins();
    AMGX_finalize();
    return 0;
}
