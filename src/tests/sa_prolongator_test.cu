// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include "unit_test.h"
#include "amg_solver.h"
#include <matrix_io.h>
#include "test_utils.h"
#include "util.h"
#include "time.h"

namespace amgx
{

// Test SA prolongator smoothing via end-to-end PCG + SA-AMG solve.
// Validates that the full SA pipeline (including P = (I - omega*D^{-1}*A)*P_tent)
// produces a working preconditioner that converges on a Poisson problem.
DECLARE_UNITTEST_BEGIN(SAProlongatorTest);

void run()
{
    randomize(42);

    // Generate a 2D 5-point Poisson matrix (10x10 = 100 nodes)
    Matrix_h A_h;
    Vector_h b_h, x_h;
    generatePoissonForTest(A_h, 1, 0, 5, 10, 10);
    int num_rows = A_h.get_num_rows();
    PrintOnFail("SA Prolongator: matrix has %d rows", num_rows);
    UNITTEST_ASSERT_TRUE(num_rows == 100);

    // RHS = ones, initial guess = zeros
    b_h.resize(num_rows, 1.0);
    x_h.resize(num_rows, 0.0);
    b_h.set_block_dimx(1);
    b_h.set_block_dimy(A_h.get_block_dimy());
    x_h.set_block_dimx(1);
    x_h.set_block_dimy(A_h.get_block_dimx());

    // Transfer to device
    MatrixA A_d = A_h;
    A_d.computeDiagonal();
    A_d.set_initialized(1);
    VVector b_d = b_h;
    VVector x_d = x_h;

    // Configure PCG with SA-AMG preconditioner using MIS selector and Chebyshev smoother
    AMG_Configuration cfg;
    UNITTEST_ASSERT_TRUE(cfg.parseParameterString(
        "{"
            "\"config_version\": 2, "
            "\"solver\": {"
                "\"solver\": \"PCG\", "
                "\"max_iters\": 100, "
                "\"convergence\": \"RELATIVE_INI\", "
                "\"tolerance\": 1e-8, "
                "\"norm\": \"L2\", "
                "\"monitor_residual\": 1, "
                "\"print_solve_stats\": 1, "
                "\"obtain_timings\": 1, "
                "\"scope\": \"main\", "
                "\"preconditioner\": {"
                    "\"solver\": \"AMG\", "
                    "\"algorithm\": \"AGGREGATION\", "
                    "\"selector\": \"MIS\", "
                    "\"mis_k\": 2, "
                    "\"mis2_algorithm\": 1, "
                    "\"aggressive_levels\": 1, "
                    "\"max_levels\": 10, "
                    "\"min_coarse_rows\": 4, "
                    "\"cycle\": \"V\", "
                    "\"presweeps\": 1, "
                    "\"postsweeps\": 1, "
                    "\"print_grid_stats\": 1, "
                    "\"smoother\": {"
                        "\"solver\": \"CHEBYSHEV\", "
                        "\"chebyshev_polynomial_order\": 1, "
                        "\"chebyshev_lambda_estimate_mode\": 4, "
                        "\"chebyshev_lmin_denom\": 11, "
                        "\"preconditioner\": {"
                            "\"solver\": \"BLOCK_JACOBI\", "
                            "\"relaxation_factor\": 1.0"
                        "}"
                    "}, "
                    "\"coarse_solver\": \"DENSE_LU_SOLVER\""
                "}"
            "}"
        "}"
    ) == AMGX_OK);

    // Create solver and run setup
    Resources res;
    AMG_Solver<TConfig> solver(&res, cfg);
    AMGX_ERROR setup_err = solver.setup(A_d);
    PrintOnFail("SA Prolongator: setup failed with error %d", (int)setup_err);
    UNITTEST_ASSERT_TRUE(setup_err == AMGX_OK);

    // Run solve
    AMGX_STATUS solve_status = AMGX_ST_CONVERGED;
    AMGX_ERROR solve_err = solver.solve(b_d, x_d, solve_status);
    PrintOnFail("SA Prolongator: solve returned error %d", (int)solve_err);
    UNITTEST_ASSERT_TRUE(solve_err == AMGX_OK);

    // Check convergence
    PrintOnFail("SA Prolongator: solve did not converge, status=%d", (int)solve_status);
    UNITTEST_ASSERT_TRUE(solve_status == AMGX_ST_CONVERGED);

    // Check iteration count is reasonable (preconditioned should converge fast)
    int num_iters = solver.get_num_iters();
    PrintOnFail("SA Prolongator: took %d iterations (expected < 50)", num_iters);
    UNITTEST_ASSERT_TRUE(num_iters > 0);
    UNITTEST_ASSERT_TRUE(num_iters < 50);
}

DECLARE_UNITTEST_END(SAProlongatorTest);

SAProlongatorTest <TemplateMode<AMGX_mode_dDDI>::Type> SAProlongatorTest_instance_mode_dDDI;

} // namespace amgx
