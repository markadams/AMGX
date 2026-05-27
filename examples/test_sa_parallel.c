// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// test_sa_parallel.c
//
// MPI-parallel test driver for SA-AMG in AMGx (Task 1.1).
//
// Generates a distributed 2D 5-point Poisson matrix (via the 7-pt generator
// with nz=pz=1, which degenerates to the 5-pt stencil in 2D).  Each MPI rank
// owns one GPU and a contiguous block of rows.
//
// Solver: PCG outer + SA-AMG preconditioner
//   - MIS selector (mis_k=1)
//   - Chebyshev smoother (polynomial order 3)
//   - chebyshev_lambda_estimate_mode=4  (reuse SA eigenvalue estimate)
//   - sa_vectors=1  (scalar problem, constant near-null vector)
//   - aggressive_levels=0
//
// Usage:
//   mpirun -n <nranks> ./test_sa_parallel [-n <grid_size>] [-mode dDDI|dDFI|dFFI]
//
//   -n <grid_size>  : side length of the 2D grid (default 100 → 10000 DOFs total)
//   -mode           : AMGx precision mode (default dDDI)
//
// Build (example on Perlmutter):
//   mpicc -O2 -I../include examples/test_sa_parallel.c \
//         -o test_sa_parallel ./libamgxsh.so -lcudart -lcublas -lcusparse -lm
//
// Expected output: "PASSED" when PCG converges within tolerance.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "cuda_runtime.h"
#include "amgx_c.h"

/* -----------------------------------------------------------------------
 * CUDA error macro
 * --------------------------------------------------------------------- */
#define CUDA_SAFE_CALL(call) do {                                          \
    cudaError_t _err = (call);                                             \
    if (_err != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                __FILE__, __LINE__, cudaGetErrorString(_err));             \
        MPI_Abort(MPI_COMM_WORLD, 1);                                      \
    }                                                                      \
} while (0)

#define MAX_MSG_LEN 4096

/* -----------------------------------------------------------------------
 * PCG + SA-AMG configuration string.
 *
 * Outer solver : PCG (max 200 iterations, relative residual tolerance 1e-8)
 * Preconditioner: AMG with AGGREGATION algorithm
 *   selector        = MIS  (mis_k=1 for parallel safety)
 *   smoother        = CHEBYSHEV_POLY (order 3)
 *   lambda_estimate = mode 4 (reuse SA Rayleigh-quotient estimate)
 *   sa_vectors      = 1  (one constant near-null vector for scalar PDE)
 *   aggressive_levels = 0
 *   coarse_solver   = DENSE_LU_SOLVER (exact solve on coarsest grid)
 * --------------------------------------------------------------------- */
static const char *PCG_SA_MIS_CONFIG =
    "{"
    "  \"config_version\": 2,"
    "  \"solver\": {"
    "    \"solver\": \"PCG\","
    "    \"max_iters\": 200,"
    "    \"convergence\": \"RELATIVE_INI\","
    "    \"tolerance\": 1e-8,"
    "    \"norm\": \"L2\","
    "    \"monitor_residual\": 1,"
    "    \"print_solve_stats\": 1,"
    "    \"obtain_timings\": 0,"
    "    \"preconditioner\": {"
    "      \"algorithm\": \"AGGREGATION\","
    "      \"solver\": \"AMG\","
    "      \"selector\": \"MIS\","
    "      \"mis_k\": 1,"
    "      \"sa_vectors\": 1,"
    "      \"aggressive_levels\": 0,"
    "      \"smoother\": {"
    "        \"solver\": \"CHEBYSHEV\","
    "        \"preconditioner\": {"
    "          \"solver\": \"BLOCK_JACOBI\","
    "          \"max_iters\": 1"
    "        },"
    "        \"max_iters\": 1,"
    "        \"chebyshev_polynomial_order\": 3,"
    "        \"chebyshev_lambda_estimate_mode\": 4,"
    "        \"chebyshev_lmin_denom\": 10.0,"
    "        \"monitor_residual\": 0,"
    "        \"print_solve_stats\": 0"
    "      },"
    "      \"presweeps\": 1,"
    "      \"postsweeps\": 1,"
    "      \"coarse_solver\": \"DENSE_LU_SOLVER\","
    "      \"max_iters\": 1,"
    "      \"monitor_residual\": 0,"
    "      \"print_solve_stats\": 0,"
    "      \"print_grid_stats\": 1,"
    "      \"min_coarse_rows\": 32,"
    "      \"max_levels\": 20,"
    "      \"cycle\": \"V\""
    "    }"
    "  }"
    "}";

/* -----------------------------------------------------------------------
 * Print callback: only rank 0 writes to stdout.
 * --------------------------------------------------------------------- */
static void print_callback(const char *msg, int length)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) { printf("%s", msg); fflush(stdout); }
}

/* -----------------------------------------------------------------------
 * Fatal error: print message and abort all ranks.
 * --------------------------------------------------------------------- */
static void fatal(const char *msg)
{
    fprintf(stderr, "ERROR: %s\n", msg);
    fflush(stderr);
    MPI_Abort(MPI_COMM_WORLD, 1);
}

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    /* --- MPI + GPU init ------------------------------------------------ */
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);

    int gpu_count = 0;
    CUDA_SAFE_CALL(cudaGetDeviceCount(&gpu_count));
    int lrank = rank % gpu_count;          /* local GPU index for this rank */
    CUDA_SAFE_CALL(cudaSetDevice(lrank));

    if (rank == 0)
        printf("test_sa_parallel: %d rank(s), %d GPU(s) visible\n",
               nranks, gpu_count);

    /* --- Parse command-line arguments ---------------------------------- */
    int grid_n = 100;                      /* default: 100×100 = 10000 DOFs */
    AMGX_Mode mode = AMGX_mode_dDDI;      /* default: double mat/vec, int idx */

    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc)
        {
            grid_n = atoi(argv[++i]);
            if (grid_n < 1) fatal("-n must be a positive integer");
        }
        else if (strcmp(argv[i], "-mode") == 0 && i + 1 < argc)
        {
            ++i;
            if      (strcmp(argv[i], "dDDI") == 0) mode = AMGX_mode_dDDI;
            else if (strcmp(argv[i], "dDFI") == 0) mode = AMGX_mode_dDFI;
            else if (strcmp(argv[i], "dFFI") == 0) mode = AMGX_mode_dFFI;
            else fatal("unknown mode; choose dDDI, dDFI, or dFFI");
        }
    }

    /* 2D grid: nx × ny points, distributed as (nranks × 1) strips.
     * The 7-pt generator with nz=pz=1 produces the 5-pt 2D stencil. */
    int nx = grid_n;          /* local x-size per rank */
    int ny = grid_n;          /* global y-size          */
    int nz = 1;               /* 2D: z-dimension = 1    */
    int px = nranks;          /* split only in x        */
    int py = 1;
    int pz = 1;
    int n_local = nx * ny * nz;   /* DOFs owned by this rank */

    if (rank == 0)
        printf("Grid: %d x %d (global %d DOFs), %d rank(s)\n",
               nx * nranks, ny, nx * nranks * ny, nranks);

    /* --- AMGx init ----------------------------------------------------- */
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    /* --- Create config from embedded string ---------------------------- */
    AMGX_config_handle cfg;
    AMGX_SAFE_CALL(AMGX_config_create(&cfg, PCG_SA_MIS_CONFIG));
    /* Enable internal exception handling so we don't need AMGX_SAFE_CALL
     * on every subsequent call. */
    AMGX_SAFE_CALL(AMGX_config_add_parameters(&cfg, "exception_handling=1"));

    /* --- Create resources, matrix, vectors, solver --------------------- */
    AMGX_resources_handle rsrc;
    AMGX_resources_create(&rsrc, cfg, &comm, 1, &lrank);

    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;

    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    /* --- Query number of halo rings required by this config ------------ */
    int nrings = 1;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);

    /* --- Generate distributed 2D Poisson (5-pt stencil via 7-pt API) -- */
    /* AMGX_generate_distributed_poisson_7pt sets up the distributed CSR
     * matrix A, and uploads b=ones, x=zeros internally.               */
    AMGX_generate_distributed_poisson_7pt(A, b, x,
                                          nrings, nrings,
                                          nx, ny, nz,
                                          px, py, pz);

    /* --- Override RHS = ones, initial guess = zeros -------------------- */
    /* The generator already sets b=1 and x=0, but we re-upload to be
     * explicit and to demonstrate the upload API.                      */
    size_t sizeof_v = (AMGX_GET_MODE_VAL(AMGX_VecPrecision, mode) == AMGX_vecDouble)
                      ? sizeof(double) : sizeof(float);

    void *b_h = malloc(n_local * sizeof_v);
    void *x_h = malloc(n_local * sizeof_v);
    if (!b_h || !x_h) fatal("malloc failed for host vectors");

    memset(x_h, 0, n_local * sizeof_v);
    for (int i = 0; i < n_local; i++)
    {
        if (sizeof_v == sizeof(double))
            ((double *)b_h)[i] = 1.0;
        else
            ((float  *)b_h)[i] = 1.0f;
    }

    /* Bind vectors to the matrix (sets up halo communication pattern). */
    AMGX_vector_bind(x, A);
    AMGX_vector_bind(b, A);
    AMGX_vector_upload(x, n_local, 1, x_h);
    AMGX_vector_upload(b, n_local, 1, b_h);

    free(b_h);
    free(x_h);

    /* --- Setup + Solve ------------------------------------------------- */
    MPI_Barrier(comm);
    AMGX_solver_setup(solver, A);

    MPI_Barrier(comm);
    AMGX_solver_solve(solver, b, x);

    /* --- Check convergence status -------------------------------------- */
    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(solver, &status);

    MPI_Barrier(comm);
    if (rank == 0)
    {
        if (status == AMGX_SOLVE_SUCCESS)
            printf("\ntest_sa_parallel: PASSED (PCG+SA-AMG converged)\n");
        else
            printf("\ntest_sa_parallel: FAILED (status=%d)\n", (int)status);
        fflush(stdout);
    }

    /* --- Cleanup ------------------------------------------------------- */
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);

    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    AMGX_SAFE_CALL(AMGX_finalize());

    MPI_Finalize();
    return (status == AMGX_SOLVE_SUCCESS) ? 0 : 1;
}
