// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#ifdef AMGX_USE_CUDSS
#include <cudss.h>
#endif

#include <solvers/solver.h>

namespace amgx
{
namespace cudss_solver
{

// Empty primary template
template <class T_Config>
class CudssSolver
{
};

// Host specialization — stub (cuDSS is GPU-only)
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CudssSolver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
    : public Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >
{
  public:
    typedef Solver<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> > Base;
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;

    CudssSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
        : Base(cfg, cfg_scope, tmng)
    {
        FatalError("CudssSolver: no host implementation", AMGX_ERR_NOT_IMPLEMENTED);
    }

    ~CudssSolver() {}

    bool isColoringNeeded() const override { return false; }
    bool getReorderColsByColorDesired() const override { return false; }
    bool getInsertDiagonalDesired() const override { return false; }
    void solver_setup(bool reuse_matrix_structure) override {}
    void solve_init(Vector<TConfig_h> &b, Vector<TConfig_h> &x, bool xIsZero) override {}
    AMGX_STATUS solve_iteration(Vector<TConfig_h> &b, Vector<TConfig_h> &x, bool xIsZero) override
    {
        return AMGX_ST_ERROR;
    }
    void solve_finalize(Vector<TConfig_h> &b, Vector<TConfig_h> &x) override {}
    void print_solver_parameters() const override {}
};

// Device specialization — actual cuDSS implementation
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
class CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
    : public Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >
{
  public:
    typedef Solver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> > Base;
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> Config_d;
    typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> Config_h;
    typedef typename Config_d::MatPrec ValueTypeA;
    typedef typename Config_d::VecPrec ValueTypeB;
    typedef typename Config_d::IndPrec IndexType;
    typedef Matrix<Config_d> Matrix_d;
    typedef Matrix<Config_h> Matrix_h;
    typedef Vector<Config_d> Vector_d;
    typedef Vector<Config_h> Vector_h;
    typedef typename Matrix_d::IVector IVector_d;
    typedef typename Matrix_h::IVector IVector_h;
    typedef typename Matrix_d::MVector MVector_d;
    typedef typename Matrix_h::MVector MVector_h;

    CudssSolver(AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng);
    ~CudssSolver();

    bool isColoringNeeded() const override { return false; }
    bool getReorderColsByColorDesired() const override { return false; }
    bool getInsertDiagonalDesired() const override { return false; }

    void solver_setup(bool reuse_matrix_structure) override;
    void solve_init(Vector_d &b, Vector_d &x, bool xIsZero) override;
    AMGX_STATUS solve_iteration(Vector_d &b, Vector_d &x, bool xIsZero) override;
    void solve_finalize(Vector_d &b, Vector_d &x) override;
    void print_solver_parameters() const override;

  private:
#ifdef AMGX_USE_CUDSS
    cudssHandle_t  m_handle;
    cudssConfig_t  m_config;
    cudssData_t    m_data;
    cudssMatrix_t  m_cudss_A;  // view of AMGx's device CSR arrays
#endif
    bool           m_factored;
    int            m_reorder;        // 0=none, 1=AMD, 2=METIS
    std::string    m_matrix_type;    // "SPD" or "GENERAL"
    bool           m_structure_set;  // tracks whether symbolic analysis has been done
    bool           m_enable_exact_solve;  // gather global matrix for distributed solve

    // Multi-GPU exact solve: cached gather data (mirrors Dense_LU_Solver pattern)
    int            m_num_rows_global;
    int            m_nnz_global;
    IVector_h      m_nz_all;         // nnz per rank
    IVector_h      m_nz_displs;      // nnz displacements
    IVector_h      m_row_all;        // rows per rank
    IVector_h      m_row_displs;     // row displacements
    IVector_d      m_Arows_global;   // gathered global row offsets (device)
    IVector_d      m_Acols_global;   // gathered global col indices (device)
    MVector_d      m_Avals_global;   // gathered global values (device)

    // Block-CSR expansion: scalar CSR arrays for cuDSS (block_size > 1)
    IVector_d      m_scalar_row_offsets;  // expanded scalar row offsets (device)
    IVector_d      m_scalar_col_indices;  // expanded scalar col indices (device)
    MVector_d      m_scalar_values;       // expanded scalar values (device)
};

// Factory class
template <class T_Config>
class CudssSolverFactory : public SolverFactory<T_Config>
{
  public:
    Solver<T_Config> *create(AMG_Config &cfg,
                              const std::string &cfg_scope,
                              ThreadManager *tmng) override
    {
        return new CudssSolver<T_Config>(cfg, cfg_scope, tmng);
    }
};

} // namespace cudss_solver
} // namespace amgx
