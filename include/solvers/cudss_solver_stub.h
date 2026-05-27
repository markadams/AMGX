// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// include/solvers/cudss_solver_stub.h
//
// Provides the CudssSolverFactory class definition when AMGX_USE_CUDSS=OFF.
// Included by core.cu so it can register "CUDSS_SOLVER" in the factory
// without pulling in <cudss.h>.
//
// When AMGX_USE_CUDSS=ON this header is NOT included; the real factory is
// defined in external/cudss/cudss_solver.cu and declared via its own header.

#pragma once

#include <solvers/solver.h>
#include <error.h>

namespace amgx
{
namespace cudss_solver
{

template <class T_Config>
class CudssSolverStub : public Solver<T_Config>
{
  public:
    typedef typename Solver<T_Config>::VVector VVector;

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

    bool isColoringNeeded()             const override { return false; }
    bool getReorderColsByColorDesired() const override { return false; }
    bool getInsertDiagonalDesired()     const override { return false; }

    void solver_setup(bool) override {}
    AMGX_STATUS solve_iteration(VVector &, VVector &, bool) override
    {
        return AMGX_ST_ERROR;
    }
};

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
