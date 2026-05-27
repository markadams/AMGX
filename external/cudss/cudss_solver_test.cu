// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause

// Unit tests for CudssSolver — requires AMGX_USE_CUDSS build flag.
// Tests follow the same pattern as src/tests/dense_lu.cu.
//
// Run with:
//   ./unit_tests --gtest_filter="CudssSolver*"

#ifdef AMGX_USE_CUDSS

#include "unit_test.h"
#include "matrix.h"
#include "blas.h"
#include "multiply.h"
#include "cudss_solver.h"
#include "test_utils.h"

namespace amgx
{

// ============================================================
// Test 1: Solve identity system (GENERAL, 32x32)
//   A = I, b = random → x should equal b exactly
// ============================================================

DECLARE_UNITTEST_BEGIN(CudssSolverTest_Solve_Id_32);

void run()
{
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type Config_h;
    typedef Vector<Config_h> FVector_h;
    typedef typename T_Config::MatPrec Matrix_data;

    const int N = 32;
    Matrix<T_Config> A(N, N, N, CSR);
    A.set_initialized(0);

    // Build identity matrix on host then copy to device
    {
        typedef typename Config_h::template setVecPrec<AMGX_vecInt>::Type IVecConfig_h;
        typedef Vector<Config_h>      FVec_h;
        typedef Vector<IVecConfig_h>  IVec_h;
        IVec_h row_offsets(N + 1), col_indices(N);
        for (int i = 0; i < N; ++i)
        {
            row_offsets[i] = i;
            col_indices[i] = i;
        }
        row_offsets[N] = N;
        FVec_h values(N, Matrix_data(1));
        A.row_offsets.copy(row_offsets);
        A.col_indices.copy(col_indices);
        A.values.copy(values);
    }

    A.set_initialized(1);

    AMG_Config cfg;
    cfg.parseParameterString("cudss_matrix_type=GENERAL, cudss_reorder=0");
    cudss_solver::CudssSolver<T_Config> solver(cfg, "default", NULL);
    solver.setup(A, false);

    // Random RHS
    FVector_h b_h(N);
    for (int i = 0; i < N; ++i)
    {
        b_h[i] = Matrix_data(rand()) / RAND_MAX;
    }

    Vector<T_Config> b(b_h), x(N);
    solver.solve(b, x, false);

    // For identity: x should equal b
    UNITTEST_ASSERT_EQUAL(x, b);
}

DECLARE_UNITTEST_END(CudssSolverTest_Solve_Id_32)

CudssSolverTest_Solve_Id_32<TemplateMode<AMGX_mode_dDDI>::Type> CudssSolverTest_Solve_Id_32_dDDI;
CudssSolverTest_Solve_Id_32<TemplateMode<AMGX_mode_dFFI>::Type> CudssSolverTest_Solve_Id_32_dFFI;

// ============================================================
// Test 2: Solve 2D Poisson (GENERAL, ~100 rows)
//   Verify residual ||Ax - b|| / ||b|| < tolerance
// ============================================================

DECLARE_UNITTEST_BEGIN(CudssSolverTest_Solve_Poisson2D_General);

void run()
{
    typedef typename T_Config::MatPrec Matrix_data;
    typedef typename T_Config::template setMemSpace<AMGX_device>::Type Config_d;
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type   Config_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<Config_d> Vector_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;

    Matrix_h A_h;
    A_h.set_initialized(0);
    // 10x10 2D Poisson → 100 rows, 5-point stencil
    generatePoissonForTest(A_h, 1, 0, 5, 10, 10);
    A_h.set_initialized(1);

    AMG_Config cfg;
    cfg.parseParameterString("cudss_matrix_type=GENERAL, cudss_reorder=1");
    Matrix_d A_d(A_h);
    cudss_solver::CudssSolver<T_Config> solver(cfg, "default", NULL);
    solver.setup(A_d, false);

    const int n = A_h.get_num_rows();
    Vector_d b_d(n), x_d(n), r_d(n);
    thrust_wrapper::fill<AMGX_device>(b_d.begin(), b_d.end(), Matrix_data(1));
    solver.solve(b_d, x_d, false);
    solver.compute_residual(b_d, x_d, r_d);

    Vector_h resid_nrm(1);
    solver.compute_norm(r_d, resid_nrm);

    if (T_Config::matPrec == AMGX_matDouble)
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0, 1.0e-10);
    }
    else
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0f, 1.0e-4f);
    }
}

DECLARE_UNITTEST_END(CudssSolverTest_Solve_Poisson2D_General)

CudssSolverTest_Solve_Poisson2D_General<TemplateMode<AMGX_mode_dDDI>::Type> CudssSolverTest_Solve_Poisson2D_General_dDDI;

// ============================================================
// Test 3: Solve 2D Poisson (SPD path — Cholesky)
//   Same problem, but use cudss_matrix_type=SPD
//   Verify residual matches GENERAL path
// ============================================================

DECLARE_UNITTEST_BEGIN(CudssSolverTest_Solve_Poisson2D_SPD);

void run()
{
    typedef typename T_Config::MatPrec Matrix_data;
    typedef typename T_Config::template setMemSpace<AMGX_device>::Type Config_d;
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type   Config_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<Config_d> Vector_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;

    Matrix_h A_h;
    A_h.set_initialized(0);
    // 10x10 2D Poisson → 100 rows, SPD
    generatePoissonForTest(A_h, 1, 0, 5, 10, 10);
    A_h.set_initialized(1);

    AMG_Config cfg;
    cfg.parseParameterString("cudss_matrix_type=SPD, cudss_reorder=1");
    Matrix_d A_d(A_h);
    cudss_solver::CudssSolver<T_Config> solver(cfg, "default", NULL);
    solver.setup(A_d, false);

    const int n = A_h.get_num_rows();
    Vector_d b_d(n), x_d(n), r_d(n);
    thrust_wrapper::fill<AMGX_device>(b_d.begin(), b_d.end(), Matrix_data(1));
    solver.solve(b_d, x_d, false);
    solver.compute_residual(b_d, x_d, r_d);

    Vector_h resid_nrm(1);
    solver.compute_norm(r_d, resid_nrm);

    if (T_Config::matPrec == AMGX_matDouble)
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0, 1.0e-10);
    }
    else
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0f, 1.0e-4f);
    }
}

DECLARE_UNITTEST_END(CudssSolverTest_Solve_Poisson2D_SPD)

CudssSolverTest_Solve_Poisson2D_SPD<TemplateMode<AMGX_mode_dDDI>::Type> CudssSolverTest_Solve_Poisson2D_SPD_dDDI;

// ============================================================
// Test 4: Reuse matrix structure (refactorization path)
//   Setup once, then call setup again with reuse_matrix_structure=true
//   Verify solution is still correct after refactorization
// ============================================================

DECLARE_UNITTEST_BEGIN(CudssSolverTest_Reuse_Structure);

void run()
{
    typedef typename T_Config::MatPrec Matrix_data;
    typedef typename T_Config::template setMemSpace<AMGX_device>::Type Config_d;
    typedef typename T_Config::template setMemSpace<AMGX_host>::Type   Config_h;
    typedef Matrix<Config_d> Matrix_d;
    typedef Vector<Config_d> Vector_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_h> Vector_h;

    Matrix_h A_h;
    A_h.set_initialized(0);
    generatePoissonForTest(A_h, 1, 0, 5, 10, 10);
    A_h.set_initialized(1);

    AMG_Config cfg;
    cfg.parseParameterString("cudss_matrix_type=GENERAL, cudss_reorder=1");
    Matrix_d A_d(A_h);
    cudss_solver::CudssSolver<T_Config> solver(cfg, "default", NULL);

    // First setup: full symbolic + numeric factorization
    solver.setup(A_d, false);

    // Second setup: reuse structure (only numeric refactorization)
    solver.setup(A_d, true);

    const int n = A_h.get_num_rows();
    Vector_d b_d(n), x_d(n), r_d(n);
    thrust_wrapper::fill<AMGX_device>(b_d.begin(), b_d.end(), Matrix_data(1));
    solver.solve(b_d, x_d, false);
    solver.compute_residual(b_d, x_d, r_d);

    Vector_h resid_nrm(1);
    solver.compute_norm(r_d, resid_nrm);

    if (T_Config::matPrec == AMGX_matDouble)
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0, 1.0e-10);
    }
    else
    {
        UNITTEST_ASSERT_EQUAL_TOL(resid_nrm[0], 0.0f, 1.0e-4f);
    }
}

DECLARE_UNITTEST_END(CudssSolverTest_Reuse_Structure)

CudssSolverTest_Reuse_Structure<TemplateMode<AMGX_mode_dDDI>::Type> CudssSolverTest_Reuse_Structure_dDDI;

} // namespace amgx

#endif // AMGX_USE_CUDSS
