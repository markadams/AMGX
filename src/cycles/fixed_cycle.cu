// SPDX-FileCopyrightText: 2013 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amg_level.h>
#include <cycles/fixed_cycle.h>
#include <blas.h>
#include <cycles/v_cycle.h>
#include <cycles/w_cycle.h>
#include <cycles/f_cycle.h>
#include <cycles/cg_cycle.h>
#include <cycles/cg_flex_cycle.h>
#include <thrust/inner_product.h>
#include <cycles/convergence_analysis.h>
#include <sstream>
#include <util.h>
#include <distributed/glue.h>

#include <amgx_types/util.h>

// ---- MG-DIAG: Diagnostic norm prints for comparing with PETSc GAMG ----
// Set AMGX_MG_DIAG=1 to enable at compile time, or always enabled for now.
#ifndef AMGX_MG_DIAG
#define AMGX_MG_DIAG 1
#endif

#if AMGX_MG_DIAG
#include <cmath>
#include <vector>
// Helper: compute 2-norm of a device vector by copying to host (avoids template linkage issues)
template <typename VVector>
static double mg_diag_norm2(const VVector &v)
{
    int n = v.size();
    if (n == 0) return 0.0;
    typedef typename VVector::value_type VT;
    typedef typename amgx::types::PODTypes<VT>::type PodT;
    std::vector<VT> h_raw(n);
    cudaMemcpy(h_raw.data(), v.raw(), n * sizeof(VT), cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        PodT a = amgx::types::util<VT>::abs(h_raw[i]);
        sum += (double)a * (double)a;
    }
    return std::sqrt(sum);
}
#endif

namespace amgx
{

template< class T_Config, template<AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec> class CycleDispatcher >
void FixedCycle<T_Config, CycleDispatcher>::cycle( AMG_Class *amg, AMG_Level<T_Config> *level, VVector &b, VVector &x )
{
    AMGX_CPU_COND_MARKER(level->isFinest(), "CYCLE", "Start new cycle");
    typedef typename VVector::value_type ValueType;
    typedef typename TConfig::MemSpace MemorySpace;
    typedef typename types::PODTypes<ValueType>::type PodType;
    Matrix<T_Config> &A = level->getA();
    Solver<T_Config> *smoother = level->getSmoother();
    VVector &bc = level->getbc();
    bc.set_block_dimx(1);
    bc.set_block_dimy(A.get_block_dimy());
    VVector &xc = level->getxc();
    xc.set_block_dimx(1);
    xc.set_block_dimy(A.get_block_dimx());
    VVector &r = level->getr();
    int levelnum = A.template getParameter <int>("level");
    int *smoothing_direction = nullptr;

#if AMGX_MG_DIAG
    // Static iteration counter — only print for first 5 outer iterations
    static int mg_diag_iter = 0;
    bool mg_diag_print = level->isFinest() && (mg_diag_iter < 5);
    if (level->isFinest()) mg_diag_iter++;
    int mg_diag_it = mg_diag_iter;  // capture current iteration number
#endif

    if (!(A.hasParameter("smoothing_direction")) )
    {
        smoothing_direction = new int;
        A.template setParameterPtr <int> ("smoothing_direction", smoothing_direction);
    }
    else
    {
        smoothing_direction = A.template getParameterPtr <int> ("smoothing_direction");
    }

    A.setView(OWNED);

#if AMGX_MG_DIAG
    if (mg_diag_print)
    {
        double nb = mg_diag_norm2(b);
        double nx = mg_diag_norm2(x);
        amgx_printf("[MG-DIAG] iter=%d level=%d ENTRY  ||b||=%.15e  ||x||=%.15e  n=%d\n",
                     mg_diag_it, levelnum, nb, nx, (int)b.size());
    }
#endif

    if (this->isASolvable(A))
    {
        this->solveExactly(A, x, b);
        return;
    }
    else
    {
        //Pre smooth
        level->Profile.tic("Smoother");
        bool xIsZero = false;

        if (level->isInitCycle())
        {
            xIsZero = true;
        }

        *smoothing_direction = 0;
        {
            AMGX_CPU_PROFILER( "FixedCycle::cycle_@presmooth" );
            int n_presweeps;

            if ( level->isCoarsest() && amg->getCoarseSolver(MemorySpace()) != NULL) // Coarsest level, with coarse solver
            {
                n_presweeps = 0;
            }
            else if (level->isCoarsest()) // coarsest level, no coarse solver
            {
                n_presweeps = amg->getNumCoarsestsweeps();
            }
            else if (level->isFinest() && amg->getNumFinestsweeps() != -1)
            {
                n_presweeps = amg->getNumPresweeps() == 0 ? 0 : amg->getNumFinestsweeps();
            }
            else
            {
                n_presweeps = amg->getNumPresweeps();

                if (amg->getNumPresweeps() != 0 && amg->getIntensiveSmoothing())
                {
                    n_presweeps = std::max(n_presweeps + levelnum - 2, 0);
                }
            }

            if ( n_presweeps > 0 )
            {
                smoother->setTolerance( 0.);
                smoother->set_max_iters( n_presweeps );
                smoother->solve( b, x, xIsZero );
            }
            else if ( xIsZero )
            {
                fill( x, types::util<ValueType>::get_zero());
            }

            level->unsetInitCycle();
        }
        level->Profile.toc("Smoother");

#if AMGX_MG_DIAG
        if (mg_diag_print)
        {
            double nx = mg_diag_norm2(x);
            amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-PRESMOOTH  ||x||=%.15e\n",
                         mg_diag_it, levelnum, nx);
        }
#endif

        if ( level->isCoarsest() && amg->getCoarseSolver(MemorySpace()) != NULL)
            // Only one level with coarse solver
        {
            level->launchCoarseSolver( amg, b, x );
        }
        else if (level->isCoarsest()) // Now at coarsest level, performed coarsest_sweeps so return
        {
            return;
        }
        else // Create data necessary for next coarser cycle
        {
            r.set_block_dimy(b.get_block_dimy());
            r.set_block_dimx(1);
            int offset, size;
            A.getOffsetAndSizeForView(OWNED, &offset, &size);
            //compute residual
            level->Profile.tic("ComputeResidual");
            axmb(A, x, b, r, offset, size);
            level->Profile.toc("ComputeResidual");

#if AMGX_MG_DIAG
            if (mg_diag_print)
            {
                double nr = mg_diag_norm2(r);
                amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-RESIDUAL  ||r||=%.15e\n",
                             mg_diag_it, levelnum, nr);
            }
#endif
            //apply restriction
            // in classical the current level is consolidated while in aggregation this is the next one.
            // Hence, in classical, given a level L, if we want to consolidate L+1 vectors (ie coarse vectors of L) we have to look at L+1 flags.
            bool consolidation_flag = false;
            bool isRootPartition_flag = false;

            if (level->isClassicalAMGLevel() && !A.is_matrix_singleGPU())  // In classical consolidation we want to use A.is_matrix_distributed(), this might be an issue when n=1
            {
                consolidation_flag = level->getNextLevel(MemorySpace())->isConsolidationLevel();
                isRootPartition_flag = level->getNextLevel(MemorySpace())->getA().manager->isRootPartition();
            }
            else if (!level->isClassicalAMGLevel() && !A.is_matrix_singleGPU())
            {
                consolidation_flag = level->isConsolidationLevel();
                isRootPartition_flag = A.manager->isRootPartition();
            }

            level->Profile.tic("restrictRes");
            level->restrictResidual(r, bc);
            level->Profile.toc("restrictRes");

#if AMGX_MG_DIAG
            if (mg_diag_print)
            {
                double nbc = mg_diag_norm2(bc);
                amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-RESTRICT  ||bc||=%.15e  bc.size=%d\n",
                             mg_diag_it, levelnum, nbc, (int)bc.size());
            }
#endif

            // we have to be very carreful with !A.is_matrix_singleGPU() by A.is_matrix_distributed().
            // In classical consolidation we want to use A.is_matrix_distributed() in order to consolidateVector / unconsolidateVector
            if (!A.is_matrix_singleGPU()  && consolidation_flag)
            {
                level->consolidateVector(bc);
                level->consolidateVector(xc);
            }

            // This should work
            if ( !( !A.is_matrix_singleGPU() && consolidation_flag && !isRootPartition_flag))
            {
                //mark the next level guess for initialization
                level->setNextInitCycle( );
                static const AMGX_VecPrecision vecPrec = T_Config::vecPrec;
                static const AMGX_MatPrecision matPrec = T_Config::matPrec;
                static const AMGX_IndPrecision indPrec = T_Config::indPrec;

                //WARNING: coarse solver might be called inside generateNextCycles routine
                if ( level->isNextCoarsest( ))
                {
                    //if the next level is the coarsest then don't dispatch an entire cycle, instead just launch a single Vfixed cycle.
                    //std::cout << "launching coarsest" << std::endl;
                    level->generateNextCycles( amg, bc, xc, V_CycleDispatcher<vecPrec, matPrec, indPrec>( ) );
                }
                else
                {
                    //solve the next level using the cycle that was passed in
                    level->generateNextCycles( amg, bc, xc, CycleDispatcher<vecPrec, matPrec, indPrec>( ) );
                }
            }

            if (!A.is_matrix_singleGPU() && consolidation_flag)
            {
                level->unconsolidateVector(xc);
            }

#if AMGX_MG_DIAG
            if (mg_diag_print)
            {
                double nxc = mg_diag_norm2(xc);
                amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-COARSE-SOLVE  ||xc||=%.15e  xc.size=%d\n",
                             mg_diag_it, levelnum, nxc, (int)xc.size());
            }
#endif

            //prolongate correction
            level->prolongateAndApplyCorrection(xc, bc, x, r);
            level->Profile.toc("proCorr");

#if AMGX_MG_DIAG
            if (mg_diag_print)
            {
                double nx = mg_diag_norm2(x);
                amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-PROLONGATE  ||x||=%.15e\n",
                             mg_diag_it, levelnum, nx);
            }
#endif
            //post smooth
            *smoothing_direction = 1;
            level->Profile.tic("Smoother");
            {
                AMGX_CPU_PROFILER( "FixedCycle::cycle_@postmooth" );
                int n_postsweeps;

                if (level->isFinest() && amg->getNumFinestsweeps() != -1)
                {
                    n_postsweeps = amg->getNumPostsweeps() == 0 ? 0 : amg->getNumFinestsweeps();
                }
                else
                {
                    n_postsweeps = amg->getNumPostsweeps();

                    if (amg->getNumPostsweeps() != 0 && amg->getIntensiveSmoothing())
                    {
                        n_postsweeps = std::max(n_postsweeps + levelnum - 2, 0);
                    }
                }

                if ( amg->m_cfg->AMG_Config::template getParameter<int>( "error_scaling", amg->m_cfg_scope ) > 3 )
                {
                    n_postsweeps = 0;
                }

                if ( n_postsweeps > 0 )
                {
                    smoother->set_max_iters( n_postsweeps );
                    smoother->setTolerance( 0.);
                    smoother->solve( b, x, false );
                }
            }
            level->Profile.toc("Smoother");

#if AMGX_MG_DIAG
            if (mg_diag_print)
            {
                double nx = mg_diag_norm2(x);
                // Also compute final residual norm for comparison with PETSc
                VVector r_diag;
                r_diag.set_block_dimy(b.get_block_dimy());
                r_diag.set_block_dimx(1);
                r_diag.resize(b.size());
                int off2, sz2;
                A.getOffsetAndSizeForView(OWNED, &off2, &sz2);
                axmb(A, x, b, r_diag, off2, sz2);
                double nr_final = mg_diag_norm2(r_diag);
                amgx_printf("[MG-DIAG] iter=%d level=%d AFTER-POSTSMOOTH  ||x||=%.15e  ||r_final||=%.15e\n",
                             mg_diag_it, levelnum, nx, nr_final);
            }
#endif

            if ( (!A.is_matrix_singleGPU()) && (!level->isClassicalAMGLevel()) && consolidation_flag )
            {
                // Note: We need to use the manager/communicator from THIS level
                //       since the manager/communicator for the NEXT level is one for the
                //       reduced set of partitions after consolidation!
                if (!level->isRootPartition())
                {
                    // bc is consolidated, data is sent from non-root to root partition
                    level->getA().manager->getComms()->send_vector_wait_all(bc);
                }
                else
                {
                    // xc is consolidated and then un-consolidated again,
                    // only the MPI send-requests from the latter step need to be waited for 
                    level->getA().manager->getComms()->send_vector_wait_all(xc);
                }
            }

        }
    } //

    AMGX_CPU_COND_MARKER(level->isFinest(), "CYCLE", "End cycle");
}

/****************************************
 * Explict instantiations
 ***************************************/

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, V_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, W_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, F_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, CG_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class FixedCycle<TemplateMode<CASE>::Type, CG_Flex_CycleDispatcher>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
AMGX_FORCOMPLEX_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE
} // namespace amgx
