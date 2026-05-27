// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <solvers/solver.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <amgx_types/util.h>

namespace amgx
{

template<class T_Config>
class GMRES_Solver : public Solver<T_Config>
{
    public:
        typedef Solver<T_Config> Base;

        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;
        typedef typename types::PODTypes<ValueTypeB>::type PodTypeB;

    private:

        int m_R;  //Iterations between restarts
        int m_krylov_size;
        ValueTypeA res_pre;
        bool no_preconditioner;
        // Preconditioner
        Solver<T_Config> *m_preconditioner;

        // Near-null space storage for SA preconditioner propagation.
        // When AMG is used as a preconditioner inside GMRES, the near-null
        // space set via setNearNullSpace() must be forwarded to the
        // preconditioner before setup() is called.  Without this, m_null_dim
        // stays 0 in the aggregation AMG level, the SA path is skipped, and
        // AMG collapses to 1 level.
        std::vector<double> m_near_null_space_data;
        int m_near_null_dim  = 0;
        int m_near_null_rows = 0;

        //allocate workspace
        std::vector<VVector> m_V_vectors;
        VVector m_Z_vector;

        //HOST WORKSPACE
        cusp::array2d<ValueTypeB, cusp::host_memory, cusp::column_major> m_H; //Hessenberg matrix
        cusp::array1d<ValueTypeB, cusp::host_memory> m_s;
        cusp::array1d<ValueTypeB, cusp::host_memory> m_cs;
        cusp::array1d<ValueTypeB, cusp::host_memory> m_sn;

    public:
        // Constructor.
        GMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope );

        // Destructor
        ~GMRES_Solver();

        // Does the solver requires the residual vector storage
        bool is_residual_needed() const { return false; }
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

        bool isColoringNeeded( ) const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); else return false; }

        void getColoringScope( std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }

        bool getReorderColsByColorDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getReorderColsByColorDesired(); return false; }

        bool getInsertDiagonalDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getInsertDiagonalDesired(); return false; }

        // Initialize the solver before running the iterations.
        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_one_iteration( VVector &b, VVector &x );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        // Finalize the solver after running the iterations.
        void solve_finalize( VVector &b, VVector &x );
};

template<class T_Config>
class GMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new GMRES_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
