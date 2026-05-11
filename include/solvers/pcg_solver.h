// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
#include <vector>

namespace amgx
{

template<class T_Config>
class PCG_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeB ValueTypeB;

    private:
        // Temporary vectors needed for the computation.
        VVector m_p, m_Ap, m_z;
        // The dot product between z and the residual.
        ValueTypeB m_r_z;
        int m_buffer_N;

        bool no_preconditioner;
        Solver<T_Config> *m_preconditioner;

        // Near-null space storage for SA preconditioner propagation.
        // When AMG is used as a preconditioner inside PCG, the near-null
        // space set via AMGX_solver_set_near_null_space() must be forwarded
        // to the preconditioner's setNearNullSpace() before setup() is called.
        // Without this, the SA prolongator smoothing step is skipped and the
        // preconditioner produces an unsmoothed P that causes divergence.
        std::vector<double> m_near_null_space_data;
        int m_near_null_dim  = 0;
        int m_near_null_rows = 0;

    public:
        // Constructor.
        PCG_Solver( AMG_Config &cfg, const std::string &cfg_scope);

        ~PCG_Solver();

        // Print the solver parameters
        void printSolverParameters() const;

        // Setup the solver
        void solver_setup(bool reuse_matrix_structure);

        // Forward near-null space to the AMG preconditioner (if any).
        // Called by AMG_Solver::setup() before solver_setup().
        void setNearNullSpace(int null_dim, int num_rows, const std::vector<double> &data) override
        {
            m_near_null_dim  = null_dim;
            m_near_null_rows = num_rows;
            m_near_null_space_data = data;
            // Also forward immediately if preconditioner already exists
            if (m_preconditioner != nullptr && !data.empty())
                m_preconditioner->setNearNullSpace(null_dim, num_rows, data);
        }

        bool isColoringNeeded() const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); return false; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }

        bool getReorderColsByColorDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getReorderColsByColorDesired(); return false; }

        bool getInsertDiagonalDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getInsertDiagonalDesired(); return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

template<class T_Config>
class PCG_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new PCG_Solver<T_Config>( cfg, cfg_scope); }
};

} // namespace amgx
