// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// src/solvers/cudss_solver_stub.cu
//
// Compiled into amgx_libs when AMGX_USE_CUDSS=OFF (the default).
// Registers "CUDSS_SOLVER" in the solver factory so that config files
// referencing it produce a clear, actionable error message rather than
// a cryptic "unknown solver" failure.
//
// When AMGX_USE_CUDSS=ON this file is excluded from the build and the
// real implementation in external/cudss/cudss_solver.cu is used instead.

#include <solvers/solver.h>
#include <error.h>

namespace amgx
{
namespace cudss_solver
{

// -----------------------------------------------------------------------
// Stub solver — always throws on construction
// -----------------------------------------------------------------------

template <class T_Config>
class CudssSolverStub : public Solver<T_Config>
{
  public:
    CudssSolverStub(AMG_Config &cfg, const std::string &cfg_scope,
                    ThreadManager *tmng)
        : Solver<T_Config>(cfg, cfg_scope, tmng)
    {
        FatalError(
            "CUDSS_SOLVER is not available: AMGX was built without cuDSS "
            "support. Rebuild with -DAMGX_USE_CUDSS=ON and ensure cuDSS "
            "is installed (see external/cudss/README.md).",
            AMGX_ERR_NOT_IMPLEMENTED);
    }

    ~CudssSolverStub() {}

    bool isColoringNeeded()          const override { return false; }
    bool getReorderColsByColorDesired() const override { return false; }
    bool getInsertDiagonalDesired()  const override { return false; }

    void solver_setup(bool) override {}
    void solve_init(Vector<T_Config> &, Vector<T_Config> &, bool) override {}
    AMGX_STATUS solve_iteration(Vector<T_Config> &, Vector<T_Config> &, bool) override
    {
        return AMGX_ST_ERROR;
    }
    void solve_finalize(Vector<T_Config> &, Vector<T_Config> &) override {}
    void print_solver_parameters() const override {}
};

// -----------------------------------------------------------------------
// Factory — creates the stub (which immediately throws)
// -----------------------------------------------------------------------

template <class T_Config>
class CudssSolverFactory : public SolverFactory<T_Config>
{
  public:
    Solver<T_Config> *create(AMG_Config &cfg,
                              const std::string &cfg_scope,
                              ThreadManager *tmng) override
    {
        return new CudssSolverStub<T_Config>(cfg, cfg_scope, tmng);
    }
};

} // namespace cudss_solver
} // namespace amgx
