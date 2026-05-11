// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// test_cross_agg.c
//
// Cross-aggregate test driver for Step 6 of mis_k_mpi_parallel_plan.md.
//
// Purpose:
//   Isolate whether the convergence gap between PETSc GAMG (~16 iters) and
//   AMGx SA-MIS2 (~82 iters) on a 100x100 Poisson problem is due to:
//     (a) aggregate quality, or
//     (b) solver machinery (smoother, prolongator smoothing, etc.)
//
//   By feeding one solver's aggregates into the other, we can run all 4
//   combinations:
//     1. AMGx solver + AMGx MIS-2 aggregates  (baseline)
//     2. AMGx solver + GAMG aggregates         (import via AMGX_IMPORT_AGGREGATES)
//     3. PETSc solver + GAMG aggregates        (baseline, run separately)
//     4. PETSc solver + AMGx aggregates        (import via petsc test, run separately)
//
// Environment variables:
//   AMGX_EXPORT_AGGREGATES=<file>  -- export AMGx level-0 aggregates to file
//   AMGX_IMPORT_AGGREGATES=<file>  -- import level-0 aggregates from file instead
//                                     of running the MIS-2 selector
//
// File format (same as PETSc export):
//   # comment lines starting with '#'
//   N          (number of fine nodes, one integer on its own line)
//   agg[0]     (one aggregate id per line, 0-based)
//   agg[1]
//   ...
//   agg[N-1]
//
// Build on Perlmutter (from build_perlmutter/):
//   /opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \
//     ../examples/test_cross_agg.c -o test_cross_agg \
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
// Run (from build_perlmutter/):
//   # Generate 100x100 Poisson matrix:
//   ./examples/generate_poisson -p 5 100 100 -o poisson100.mtx
//
//   # Combination 1: AMGx solver + AMGx MIS-2 aggregates (baseline)
//   AMGX_EXPORT_AGGREGATES=amgx_aggs.txt ./test_cross_agg poisson100.mtx
//
//   # Combination 2: AMGx solver + GAMG aggregates (import)
//   AMGX_IMPORT_AGGREGATES=gamg_aggs.txt ./test_cross_agg poisson100.mtx
//
// See scripts/run_cross_agg_test.sh for the full 4-combination test.

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
// SA Cheby-2+Jacobi, lambda_mode=4, MIS-2 selector, max_levels=2.
// Two-level hierarchy: fine grid + one coarse grid solved with DENSE_LU.
// This matches the PETSc GAMG setup: aggressive_coarsening=1 (one level
// of aggressive coarsening), then direct solve on the coarse grid.
//
// max_levels=2 forces exactly one coarsening step.
// min_coarse_rows=2 ensures we always coarsen (don't bail out early).
//
// Step 7.4.2: omega changed from 1.4 to 4/3 in aggregation_amg_level.cu.
// Step 7.4.4: tolerance changed from 1e-5 to 1e-8 to match PETSc.
// -----------------------------------------------------------------------
static const char *MIS2_2LEVEL_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"CHEBYSHEV\","
    "      \"preconditioner\": {"
    "        \"solver\": \"BLOCK_JACOBI\","
    "        \"max_iters\": 1"
    "      },"
    "      \"max_iters\": 1,"
    "      \"chebyshev_polynomial_order\": 2,"
    "      \"chebyshev_lambda_estimate_mode\": 4,"
    "      \"chebyshev_lmin_denom\": 10.0,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 1,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"MIS\","
    "    \"mis_k\": 2,"
    "    \"aggressive_levels\": 1,"
    "    \"merge_singletons\": 1,"
    "    \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 2,"
    "    \"max_levels\": 2,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Step 7.4.3 (corrected in §7.7): Same as MIS2_2LEVEL_CONFIG but with
// lambda_mode=3 and PETSc's actual Chebyshev interval forced directly.
//
// PETSc GAMG sets (via gamg.c:935-936, bypassing the API):
//   cheb->emin_provided = emax / 20  (= 1.96997/20 = 0.0985)
//   cheb->emax_provided = emax       (= 1.96997, from SA power iteration)
//
// The Chebyshev transform [0, 0.1; 0, 1.1] then gives the actual interval:
//   lmin = 0*emin_provided + 0.1*emax_provided = 0.1 * 1.96997 = 0.196997
//   lmax = 0*emin_provided + 1.1*emax_provided = 1.1 * 1.96997 = 2.16697
//
// For AMGx lambda_mode=3, cheby_max_lambda/cheby_min_lambda are used
// DIRECTLY as the Chebyshev interval (no transform applied).  So we must
// use the POST-TRANSFORM targets [0.196997, 2.16697], NOT the pre-transform
// provided values [0.0985, 1.96997].
//
// Confirmed by -ksp_view output (lines printed in this order):
//   "eigenvalue targets used: min 0.196997, max 2.16697"   ← what Cheby uses
//   "eigenvalues provided (min 0.0984987, max 1.96997) with transform: [0. 0.1; 0. 1.1]"
// -----------------------------------------------------------------------
static const char *MIS2_2LEVEL_PETSC_EIG_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"CHEBYSHEV\","
    "      \"preconditioner\": {"
    "        \"solver\": \"BLOCK_JACOBI\","
    "        \"max_iters\": 1"
    "      },"
    "      \"max_iters\": 1,"
    "      \"chebyshev_polynomial_order\": 2,"
    "      \"chebyshev_lambda_estimate_mode\": 3,"
    "      \"cheby_max_lambda\": 2.16697,"
    "      \"cheby_min_lambda\": 0.196997,"
    "      \"chebyshev_lmin_denom\": 10.0,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 1,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"MIS\","
    "    \"mis_k\": 2,"
    "    \"aggressive_levels\": 1,"
    "    \"merge_singletons\": 1,"
    "    \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 2,"
    "    \"max_levels\": 2,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Step 7.8: PCG outer solver with AMG-V-cycle preconditioner.
// This matches PETSc GAMG: CG as the outer Krylov accelerator, with the
// AMG V-cycle used as a preconditioner (not as the standalone iteration).
//
// Structure (mirrors PCG_SA.json):
//   outer solver: PCG
//   preconditioner: AMG (AGGREGATION, SA, MIS-2, 2-level, Jacobi smoother)
//
// The AMG preconditioner runs exactly 1 V-cycle per PCG iteration
// (max_iters=1 in the preconditioner scope).
//
// CRITICAL: PCG requires an SPD preconditioner.  Chebyshev is asymmetric
// (forward polynomial sweeps only), so using it as the smoother makes the
// V-cycle non-SPD and PCG diverges (residual grows from iter 4 onward).
// We use BLOCK_JACOBI (symmetric) instead — matching PCG_SA.json.
//
// The near-null space propagation fix in pcg_solver.cu ensures the SA path
// runs (m_null_dim > 0) so a smoothed prolongator is built.
// -----------------------------------------------------------------------
static const char *PCG_MIS2_2LEVEL_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"solver\": \"PCG\","
    "    \"preconditioner\": {"
    "      \"algorithm\": \"AGGREGATION\","
    "      \"solver\": \"AMG\","
    "      \"smoother\": {"
    "        \"solver\": \"BLOCK_JACOBI\","
    "        \"relaxation_factor\": 0.8,"
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
    "      \"cycle\": \"V\","
    "      \"min_coarse_rows\": 2,"
    "      \"max_levels\": 2,"
    "      \"print_solve_stats\": 0,"
    "      \"print_grid_stats\": 1,"
    "      \"monitor_residual\": 0"
    "    },"
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"print_solve_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// §7.9: Jacobi-L1 Richardson smoother experiment.
//
// Undamped Richardson iteration with L1-row-norm Jacobi preconditioning:
//   x_{k+1} = x_k + D_L1^{-1} (b - A x_k)
//   (D_L1)_ii = sum_j |A_ij|   (L1 row norm, including diagonal)
//
// This is the simplest possible smoother — no eigenvalue estimation,
// no Chebyshev polynomial, no damping parameter to tune.
//
// Matches PETSc options:
//   -mg_levels_ksp_type richardson
//   -mg_levels_pc_type jacobi
//   -mg_levels_pc_jacobi_type rowl1
//   -mg_levels_pc_jacobi_rowl1_scale 1
//   -mg_levels_pc_jacobi_fixdiagonal
//
// AMGx JACOBI_L1 with relaxation_factor=1.0:
//   compute_d_kernel: d_i = sum_j |A_ij|  (L1 row norm)  ← matches rowl1 scale=1
//   jacobi_l1_postsmooth: x += omega*(b-Ax)/d  with omega=1.0  ← undamped
//
// Note: relaxation_factor default is 0.9 — must explicitly set 1.0 for undamped.
// -----------------------------------------------------------------------
static const char *JACOBI_L1_2LEVEL_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"algorithm\": \"AGGREGATION\","
    "    \"solver\": \"AMG\","
    "    \"smoother\": {"
    "      \"solver\": \"JACOBI_L1\","
    "      \"relaxation_factor\": 1.0,"
    "      \"max_iters\": 1,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0"
    "    },"
    "    \"presweeps\": 1,"
    "    \"postsweeps\": 1,"
    "    \"selector\": \"MIS\","
    "    \"mis_k\": 2,"
    "    \"aggressive_levels\": 1,"
    "    \"merge_singletons\": 1,"
    "    \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"cycle\": \"V\","
    "    \"min_coarse_rows\": 2,"
    "    \"max_levels\": 2,"
    "    \"print_solve_stats\": 1,"
    "    \"print_grid_stats\": 1,"
    "    \"monitor_residual\": 1,"
    "    \"obtain_timings\": 0"
    "  }"
    "}";

// -----------------------------------------------------------------------
// Run one solve.  label is printed as a header.
// matrix_file: path to MatrixMarket (.mtx) file.
// config_str:  JSON config string.
// null_dim:    if > 0, set near-null space to all-ones vector of this dim.
// Returns the number of iterations taken (or -1 on error).
// -----------------------------------------------------------------------
static int run_solve(const char *label,
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
    AMGX_SOLVE_STATUS     status;

    AMGX_config_create(&cfg, config_str);
    AMGX_resources_create_simple(&res, cfg);
    AMGX_matrix_create(&mtx, res, AMGX_mode_dDDI);
    AMGX_vector_create(&b,   res, AMGX_mode_dDDI);
    AMGX_vector_create(&x,   res, AMGX_mode_dDDI);
    AMGX_solver_create(&slv, res, AMGX_mode_dDDI, cfg);

    // Read matrix (and rhs/solution if present in file)
    AMGX_read_system(mtx, b, x, matrix_file);

    // Get matrix size for null space / zero initial guess
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

    // Setup solver (runs hierarchy construction, including aggregate export/import)
    AMGX_solver_setup(slv, mtx);

    AMGX_solver_solve(slv, b, x);
    AMGX_solver_get_status(slv, &status);

    int iters = -1;
    AMGX_config_get_default_number_of_rings(cfg, &iters);  // placeholder
    // The iteration count is printed by print_solve_stats=1 in the config.
    // We report convergence status here.
    printf("[CROSS-AGG] Solve status: %s\n",
           status == AMGX_SOLVE_SUCCESS ? "CONVERGED" :
           status == AMGX_SOLVE_DIVERGED ? "DIVERGED" : "NOT_CONVERGED");

    AMGX_solver_destroy(slv);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(mtx);
    AMGX_resources_destroy(res);
    AMGX_config_destroy(cfg);

    return (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix_file.mtx>\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Cross-aggregate test: runs AMGx SA-MIS2 (2-level) on the given matrix.\n");
        fprintf(stderr, "Control aggregate source via environment variables:\n");
        fprintf(stderr, "  AMGX_EXPORT_AGGREGATES=<file>  export AMGx aggregates to file\n");
        fprintf(stderr, "  AMGX_IMPORT_AGGREGATES=<file>  import aggregates from file\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Generate a 100x100 Poisson matrix:\n");
        fprintf(stderr, "  ./examples/generate_poisson -p 5 100 100 -o poisson100.mtx\n");
        return 1;
    }

    const char *matrix_file = argv[1];
    const char *import_file = getenv("AMGX_IMPORT_AGGREGATES");
    const char *export_file = getenv("AMGX_EXPORT_AGGREGATES");

    printf("=== AMGx Cross-Aggregate Test ===\n");
    printf("Matrix file: %s\n", matrix_file);
    if (import_file)
        printf("Importing aggregates from: %s\n", import_file);
    else
        printf("Using AMGx MIS-2 aggregates (may export to: %s)\n",
               export_file ? export_file : "(not exporting)");

    AMGX_initialize();
    AMGX_initialize_plugins();
    AMGX_register_print_callback(&print_callback);
    AMGX_install_signal_handler();

    // Determine label based on aggregate source
    const char *agg_src = import_file ? import_file : "AMGx-MIS2";
    char label[512];

    // --- Run 1: default config (lambda_mode=4, omega=4/3, tol=1e-8) ---
    snprintf(label, sizeof(label),
             "AMGx SA [lambda_mode=4, omega=4/3, tol=1e-8] aggs=%s",
             agg_src);
    int rc = run_solve(label, matrix_file, MIS2_2LEVEL_CONFIG, /*null_dim=*/1);

    // --- Run 2: PETSc eigenvalue bounds (lambda_mode=3, forced eig, tol=1e-8) ---
    snprintf(label, sizeof(label),
             "AMGx SA [lambda_mode=3, PETSc-eig forced, tol=1e-8] aggs=%s",
             agg_src);
    int rc2 = run_solve(label, matrix_file, MIS2_2LEVEL_PETSC_EIG_CONFIG, /*null_dim=*/1);

    // --- Run 3: PCG outer solver + AMG-V-cycle preconditioner (matches PETSc CG+GAMG) ---
    snprintf(label, sizeof(label),
             "AMGx PCG+AMG [CG outer, V-cycle prec, tol=1e-8] aggs=%s",
             agg_src);
    int rc3 = run_solve(label, matrix_file, PCG_MIS2_2LEVEL_CONFIG, /*null_dim=*/1);

    // --- Run 4: Jacobi-L1 Richardson (undamped, matches PETSc rowl1 smoother) ---
    // §7.9: Freeze aggregates + use identical simple smoother to isolate
    // prolongator construction differences between AMGx SA and PETSc GAMG.
    snprintf(label, sizeof(label),
             "AMGx SA [Jacobi-L1 omega=1, Richardson, tol=1e-8] aggs=%s",
             agg_src);
    int rc4 = run_solve(label, matrix_file, JACOBI_L1_2LEVEL_CONFIG, /*null_dim=*/1);

    AMGX_finalize_plugins();
    AMGX_finalize();
    return (rc || rc2 || rc3 || rc4) ? 1 : 0;
}
