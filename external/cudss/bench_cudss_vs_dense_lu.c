// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// bench_cudss_vs_dense_lu.c
//
// Performance comparison: CUDSS_SOLVER vs DENSE_LU_SOLVER as the AMG coarse
// solver in a PCG + SA-AMG preconditioner on a 2D 5-point Poisson problem.
//
// Both solvers are exact direct solvers on the coarsest grid.  This benchmark
// measures:
//   - AMG setup time  (AMGX_solver_setup)
//   - PCG solve time  (AMGX_solver_solve)
//   - Total time      (setup + solve)
//   - Iteration count (from AMGX_solver_get_iterations_number)
//
// Usage:
//   ./bench_cudss_vs_dense_lu [-n <grid_size>] [-mode dDDI|dDFI|dFFI]
//
//   -n <grid_size>  : side length of the 2D grid (default 200 → 40000 DOFs)
//   -mode           : AMGx precision mode (default dDDI)
//
// Build (example, single-GPU, no MPI):
//   gcc -O2 -I../include examples/bench_cudss_vs_dense_lu.c \
//       -o bench_cudss_vs_dense_lu ./libamgxsh.so -lcudart -lm
//
// Expected output: both solvers converge in the same number of iterations;
// CUDSS_SOLVER should be faster for large coarse grids (>500 rows).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "amgx_c.h"

/* -----------------------------------------------------------------------
 * Wall-clock timer (seconds)
 * --------------------------------------------------------------------- */
static double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* -----------------------------------------------------------------------
 * Print callback: write directly to stdout.
 * --------------------------------------------------------------------- */
static void print_callback(const char *msg, int length)
{
    (void)length;
    printf("%s", msg);
    fflush(stdout);
}

/* -----------------------------------------------------------------------
 * Build a PCG + SA-AMG config string with the given coarse_solver name.
 * Extra JSON key-value pairs (e.g. cudss options) go in extra_kv.
 * --------------------------------------------------------------------- */
static void build_config(char *buf, size_t bufsz,
                         const char *coarse_solver,
                         const char *extra_kv)
{
    snprintf(buf, bufsz,
        "{"
        "  \"config_version\": 2,"
        "  \"solver\": {"
        "    \"solver\": \"PCG\","
        "    \"max_iters\": 200,"
        "    \"convergence\": \"RELATIVE_INI\","
        "    \"tolerance\": 1e-8,"
        "    \"norm\": \"L2\","
        "    \"monitor_residual\": 1,"
        "    \"print_solve_stats\": 0,"
        "    \"obtain_timings\": 0,"
        "    \"preconditioner\": {"
        "      \"algorithm\": \"AGGREGATION\","
        "      \"solver\": \"AMG\","
        "      \"selector\": \"MIS\","
        "      \"mis_k\": 2,"
        "      \"sa_vectors\": 1,"
        "      \"aggressive_levels\": 1,"
        "      \"smoother\": {"
        "        \"solver\": \"CHEBYSHEV\","
        "        \"preconditioner\": {"
        "          \"solver\": \"BLOCK_JACOBI\","
        "          \"max_iters\": 1"
        "        },"
        "        \"max_iters\": 1,"
        "        \"chebyshev_polynomial_order\": 1,"
        "        \"chebyshev_lambda_estimate_mode\": 4,"
        "        \"chebyshev_lmin_denom\": 11.0,"
        "        \"monitor_residual\": 0,"
        "        \"print_solve_stats\": 0"
        "      },"
        "      \"presweeps\": 1,"
        "      \"postsweeps\": 1,"
        "      \"coarse_solver\": \"%s\","
        "      %s"
        "      \"max_iters\": 1,"
        "      \"monitor_residual\": 0,"
        "      \"print_solve_stats\": 0,"
        "      \"print_grid_stats\": 0,"
        "      \"min_coarse_rows\": 32,"
        "      \"max_levels\": 20,"
        "      \"cycle\": \"V\""
        "    }"
        "  }"
        "}",
        coarse_solver, extra_kv);
}

/* -----------------------------------------------------------------------
 * Run one benchmark trial.
 * Returns 0 on success, 1 on failure.
 * --------------------------------------------------------------------- */
static int run_trial(const char *label,
                     const char *config_str,
                     AMGX_Mode mode,
                     int n_local,
                     double *t_setup_out,
                     double *t_solve_out,
                     int    *iters_out)
{
    AMGX_config_handle   cfg    = NULL;
    AMGX_resources_handle rsrc  = NULL;
    AMGX_matrix_handle   A      = NULL;
    AMGX_vector_handle   b      = NULL;
    AMGX_vector_handle   x      = NULL;
    AMGX_solver_handle   solver = NULL;
    int rc = 0;

    AMGX_SAFE_CALL(AMGX_config_create(&cfg, config_str));
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    int device = 0;
    AMGX_resources_create_simple(&rsrc, cfg);

    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    /* Generate 2D Poisson (5-pt via 7-pt API with nz=pz=1) */
    int nx = n_local, ny = n_local, nz = 1;
    int px = 1, py = 1, pz = 1;
    AMGX_generate_distributed_poisson_7pt(A, b, x,
                                          1, 1,
                                          nx, ny, nz,
                                          px, py, pz);

    /* Upload b=1, x=0 */
    size_t sizeof_v = (AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)
                      ? sizeof(double) : sizeof(float);
    int n_dofs = nx * ny * nz;
    void *b_h = malloc(n_dofs * sizeof_v);
    void *x_h = calloc(n_dofs, sizeof_v);
    if (!b_h || !x_h) { fprintf(stderr, "malloc failed\n"); rc = 1; goto cleanup; }

    for (int i = 0; i < n_dofs; i++)
    {
        if (sizeof_v == sizeof(double)) ((double *)b_h)[i] = 1.0;
        else                            ((float  *)b_h)[i] = 1.0f;
    }
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    AMGX_vector_upload(x, n_dofs, 1, x_h);
    AMGX_vector_upload(b, n_dofs, 1, b_h);
    free(b_h); b_h = NULL;
    free(x_h); x_h = NULL;

    /* Setup */
    double t0 = wall_time();
    AMGX_solver_setup(solver, A);
    double t1 = wall_time();
    *t_setup_out = t1 - t0;

    /* Solve */
    t0 = wall_time();
    AMGX_solver_solve(solver, b, x);
    t1 = wall_time();
    *t_solve_out = t1 - t0;

    /* Check status */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);
    AMGX_solver_get_iterations_number(solver, iters_out);

    if (status != AMGX_SOLVE_SUCCESS)
    {
        fprintf(stderr, "[%s] FAILED (status=%d)\n", label, (int)status);
        rc = 1;
    }

cleanup:
    if (solver) AMGX_solver_destroy(solver);
    if (x)      AMGX_vector_destroy(x);
    if (b)      AMGX_vector_destroy(b);
    if (A)      AMGX_matrix_destroy(A);
    if (rsrc)   AMGX_resources_destroy(rsrc);
    if (cfg)    AMGX_config_destroy(cfg);
    return rc;
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    int n = 200;                       /* default: 200×200 = 40000 DOFs */
    AMGX_Mode mode = AMGX_mode_dDDI;

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
            n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc)
        {
            ++i;
            if      (strcmp(argv[i], "dDDI") == 0) mode = AMGX_mode_dDDI;
            else if (strcmp(argv[i], "dDFI") == 0) mode = AMGX_mode_dDFI;
            else if (strcmp(argv[i], "dFFI") == 0) mode = AMGX_mode_dFFI;
            else { fprintf(stderr, "Unknown mode: %s\n", argv[i]); return 1; }
        }
    }

    printf("bench_cudss_vs_dense_lu: grid %d×%d = %d DOFs, mode=%s\n",
           n, n, n * n,
           (mode == AMGX_mode_dDDI) ? "dDDI" :
           (mode == AMGX_mode_dDFI) ? "dDFI" : "dFFI");

    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    char cfg_dense[4096], cfg_cudss[4096];
    build_config(cfg_dense, sizeof(cfg_dense), "DENSE_LU_SOLVER", "");
    build_config(cfg_cudss, sizeof(cfg_cudss), "CUDSS_SOLVER",
                 "\"cudss_matrix_type\": \"SPD\", \"cudss_reorder\": 1,");

    double t_setup_dense = 0, t_solve_dense = 0;
    double t_setup_cudss = 0, t_solve_cudss = 0;
    int iters_dense = 0, iters_cudss = 0;
    int rc = 0;

    printf("\n--- DENSE_LU_SOLVER ---\n");
    rc |= run_trial("DENSE_LU", cfg_dense, mode, n,
                    &t_setup_dense, &t_solve_dense, &iters_dense);

    printf("\n--- CUDSS_SOLVER ---\n");
    rc |= run_trial("CUDSS", cfg_cudss, mode, n,
                    &t_setup_cudss, &t_solve_cudss, &iters_cudss);

    AMGX_SAFE_CALL(AMGX_finalize());

    /* Summary table */
    printf("\n");
    printf("=== Performance Comparison: %d×%d 2D Poisson ===\n", n, n);
    printf("%-20s %10s %10s %10s %8s\n",
           "Solver", "Setup(s)", "Solve(s)", "Total(s)", "Iters");
    printf("%-20s %10.4f %10.4f %10.4f %8d\n",
           "DENSE_LU_SOLVER",
           t_setup_dense, t_solve_dense, t_setup_dense + t_solve_dense,
           iters_dense);
    printf("%-20s %10.4f %10.4f %10.4f %8d\n",
           "CUDSS_SOLVER",
           t_setup_cudss, t_solve_cudss, t_setup_cudss + t_solve_cudss,
           iters_cudss);

    if (t_setup_dense > 0 && t_setup_cudss > 0)
    {
        double speedup_setup = t_setup_dense / t_setup_cudss;
        double speedup_solve = t_solve_dense / t_solve_cudss;
        double speedup_total = (t_setup_dense + t_solve_dense)
                             / (t_setup_cudss + t_solve_cudss);
        printf("\nSpeedup (DENSE_LU / CUDSS):\n");
        printf("  Setup: %.2fx   Solve: %.2fx   Total: %.2fx\n",
               speedup_setup, speedup_solve, speedup_total);
    }

    if (iters_dense != iters_cudss)
        printf("\nWARNING: iteration counts differ (%d vs %d)\n",
               iters_dense, iters_cudss);
    else
        printf("\nIteration counts match (%d) — both solvers are exact.\n",
               iters_dense);

    return rc;
}
