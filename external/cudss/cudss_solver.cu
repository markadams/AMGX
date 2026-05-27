// SPDX-FileCopyrightText: 2025 NVIDIA CORPORATION. All Rights Reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <string>
#include "cudss_solver.h"
#include <amgx_types/util.h>

#ifdef AMGX_USE_CUDSS
#include <cudss.h>
#endif

#include <amg_config.h>
#include <blas.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>

namespace amgx
{
namespace cudss_solver
{

// ============================================================
// Helper kernels for distributed (multi-GPU) exact coarse solve
// ============================================================

// Offset local row offsets to global row offsets
template <class IndexType>
__global__ void local_row_offsets_to_global(
    int num_rows, int offset, IndexType *local_Arows)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_rows) { return; }
    local_Arows[i] += offset;
}

// Offset local packed column indices to global unpacked indices
template <class IndexType, class L2GType>
__global__ void local_col_indices_to_global(
    int nnz, int num_rows, int offset,
    IndexType *local_Acols, L2GType *l2g)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= nnz) { return; }

    if (local_Acols[i] >= num_rows)
    {
        // Halo column: map to global index via local-to-global map
        local_Acols[i] = static_cast<IndexType>(l2g[local_Acols[i] - num_rows]);
    }
    else
    {
        // Owned column: offset to global position
        local_Acols[i] += offset;
    }
}

// Kernel 1: compute scalar row offsets from block-CSR row offsets.
// Each scalar row sr = block_row * bm + r has
//   (block_row_offsets[block_row+1] - block_row_offsets[block_row]) * bn entries.
// Launch with at least n_scalar+1 threads.
template <class IndexType>
__global__ void block_csr_row_offsets_kernel(
    int n_block,
    int bm,
    int bn,
    const IndexType *__restrict__ block_row_offsets,  // length n_block+1
    IndexType       *__restrict__ scalar_row_offsets) // length n_scalar+1 (output)
{
    int sr = blockIdx.x * blockDim.x + threadIdx.x;
    int n_scalar = n_block * bm;
    if (sr > n_scalar) { return; }

    if (sr == n_scalar)
    {
        // Guard value: total scalar nnz = total block nnz * bm * bn
        scalar_row_offsets[n_scalar] = block_row_offsets[n_block] * bm * bn;
    }
    else
    {
        int block_row     = sr / bm;
        int block_nnz_row = block_row_offsets[block_row + 1] - block_row_offsets[block_row];
        scalar_row_offsets[sr] = block_row_offsets[block_row] * bm * bn
                               + (sr % bm) * block_nnz_row * bn;
    }
}

// Second kernel: fill scalar col_indices and values.
// One thread per scalar nonzero.
template <class IndexType, class ValueType>
__global__ void block_csr_fill_cols_vals_kernel(
    int n_block,
    int bm,
    int bn,
    const IndexType *__restrict__ block_row_offsets,
    const IndexType *__restrict__ block_col_indices,
    const ValueType *__restrict__ block_values,
    const IndexType *__restrict__ scalar_row_offsets,
    IndexType       *__restrict__ scalar_col_indices,
    ValueType       *__restrict__ scalar_values)
{
    // Each thread handles one scalar nonzero.
    // scalar nonzero index = scalar_row_offsets[sr] + local_idx
    // We iterate over block-rows and within each block-row over scalar rows.
    int sr = blockIdx.x * blockDim.x + threadIdx.x;
    int n_scalar = n_block * bm;
    if (sr >= n_scalar) { return; }

    int block_row = sr / bm;
    int r         = sr % bm;  // scalar row within block

    int blk_start = block_row_offsets[block_row];
    int blk_end   = block_row_offsets[block_row + 1];
    int scalar_out = scalar_row_offsets[sr];

    for (int j = blk_start; j < blk_end; ++j)
    {
        int block_col = block_col_indices[j];
        for (int c = 0; c < bn; ++c)
        {
            scalar_col_indices[scalar_out] = block_col * bn + c;
            scalar_values[scalar_out]      = block_values[j * bm * bn + r * bn + c];
            ++scalar_out;
        }
    }
}

#ifdef AMGX_USE_CUDSS

// Helper: map C++ type to cudaDataType for cuDSS
template <typename T> cudaDataType cudss_data_type();
template <> cudaDataType cudss_data_type<float>()  { return CUDA_R_32F; }
template <> cudaDataType cudss_data_type<double>() { return CUDA_R_64F; }

// Helper: check cuDSS status
#define CUDSS_CHECK(call)                                                      \
    do {                                                                       \
        cudssStatus_t status = (call);                                         \
        if (status != CUDSS_STATUS_SUCCESS)                                    \
        {                                                                      \
            FatalError(std::string("cuDSS error: ") + std::to_string(status), \
                       AMGX_ERR_INTERNAL);                                     \
        }                                                                      \
    } while (0)

#endif // AMGX_USE_CUDSS

// ============================================================
// Constructor
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::CudssSolver(
    AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng)
    : Solver<Config_d>(cfg, cfg_scope, tmng),
      m_factored(false),
      m_reorder(1),
      m_matrix_type("GENERAL"),
      m_structure_set(false),
      m_enable_exact_solve(false),
      m_num_rows_global(0),
      m_nnz_global(0)
{
#ifdef AMGX_USE_CUDSS
    CUDSS_CHECK(cudssCreate(&m_handle));
    CUDSS_CHECK(cudssConfigCreate(&m_config));
    CUDSS_CHECK(cudssDataCreate(m_handle, &m_data));
    m_cudss_A = nullptr;

    // Read configuration parameters
    m_reorder     = cfg.getParameter<int>("cudss_reorder", cfg_scope);
    m_matrix_type = cfg.getParameter<std::string>("cudss_matrix_type", cfg_scope);

    // Determine if exact (global) coarse solve is enabled for distributed mode
    m_enable_exact_solve = (cfg.getParameter<int>("exact_coarse_solve", cfg_scope) == 1);

    // Direct solver: exactly 1 iteration
    this->set_max_iters(1);
#else
    FatalError("AMGx was not built with cuDSS support (AMGX_USE_CUDSS=OFF)",
               AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

// ============================================================
// Destructor
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::~CudssSolver()
{
#ifdef AMGX_USE_CUDSS
    if (m_cudss_A != nullptr)
    {
        cudssMatrixDestroy(m_cudss_A);
        m_cudss_A = nullptr;
    }
    cudssDataDestroy(m_handle, m_data);
    cudssConfigDestroy(m_config);
    cudssDestroy(m_handle);
#endif
}

// ============================================================
// print_solver_parameters
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::print_solver_parameters() const
{
    std::cout << "cudss_reorder=" << m_reorder
              << " cudss_matrix_type=" << m_matrix_type << std::endl;
}

// ============================================================
// solve_init — no-op (factorization is done in solver_setup)
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::solve_init(
    Vector_d &b, Vector_d &x, bool xIsZero)
{
    // No-op: factorization done in solver_setup
}

// ============================================================
// solve_finalize — no-op
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::solve_finalize(
    Vector_d &b, Vector_d &x)
{
    // No-op
}

// ============================================================
// solve_iteration — triangular solve via cuDSS
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
AMGX_STATUS CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::solve_iteration(
    Vector_d &b, Vector_d &x, bool xIsZero)
{
#ifdef AMGX_USE_CUDSS
    if (!m_factored)
    {
        FatalError("CudssSolver::solve_iteration called before solver_setup completed factorization",
                   AMGX_ERR_BAD_PARAMETERS);
    }

    Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);

    if (A->is_matrix_distributed() && m_enable_exact_solve)
    {
        // ---- Distributed exact solve path ----
        // Gather local RHS from all ranks, solve the global system redundantly
        // on every rank, then extract the local portion of the solution.
#ifdef AMGX_WITH_MPI
        int offset, num_rows;
        A->getOffsetAndSizeForView(OWNED, &offset, &num_rows);
        int rank = A->manager->global_id();

        // Copy local RHS to host
        MVector_h rhs_local_h(num_rows);
        amgx::thrust::copy(b.begin(), b.begin() + num_rows, rhs_local_h.begin());

        // Gather RHS from all ranks
        MVector_h rhs_global_h(m_num_rows_global);
        A->manager->getComms()->all_gather_v(
            rhs_local_h, num_rows, rhs_global_h, m_row_all, m_row_displs);

        // Copy global RHS to device
        MVector_d rhs_global_d(m_num_rows_global);
        amgx::thrust::copy(rhs_global_h.begin(), rhs_global_h.end(), rhs_global_d.begin());

        // Allocate global solution on device
        MVector_d x_global_d(m_num_rows_global);

        // Create dense vector descriptors for global RHS and solution
        cudssMatrix_t cudss_b = nullptr;
        cudssMatrix_t cudss_x = nullptr;

        CUDSS_CHECK(cudssMatrixCreateDn(
            &cudss_b, m_num_rows_global, 1, m_num_rows_global,
            rhs_global_d.raw(), cudss_data_type<ValueTypeB>(),
            CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_CHECK(cudssMatrixCreateDn(
            &cudss_x, m_num_rows_global, 1, m_num_rows_global,
            x_global_d.raw(), cudss_data_type<ValueTypeB>(),
            CUDSS_LAYOUT_COL_MAJOR));

        // Solve: x_global = A_global^{-1} * rhs_global
        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_SOLVE, m_config, m_data,
                                 m_cudss_A, cudss_x, cudss_b));

        cudssMatrixDestroy(cudss_b);
        cudssMatrixDestroy(cudss_x);

        // Extract local portion of the global solution back into x
        amgx::thrust::copy(
            x_global_d.begin() + m_row_displs[rank],
            x_global_d.begin() + m_row_displs[rank] + num_rows,
            x.begin());
#endif
    }
    else
    {
        // ---- Single-GPU or inexact (local block) solve path ----
        int n          = this->m_A->get_num_rows();
        int block_size = this->m_A->get_block_dimx();
        int n_scalar   = n * block_size;

        // Create dense vector descriptors (zero-copy: point at AMGx device arrays)
        cudssMatrix_t cudss_b = nullptr;
        cudssMatrix_t cudss_x = nullptr;

        CUDSS_CHECK(cudssMatrixCreateDn(
            &cudss_b, n_scalar, 1, n_scalar,
            b.raw(), cudss_data_type<ValueTypeB>(),
            CUDSS_LAYOUT_COL_MAJOR));

        CUDSS_CHECK(cudssMatrixCreateDn(
            &cudss_x, n_scalar, 1, n_scalar,
            x.raw(), cudss_data_type<ValueTypeB>(),
            CUDSS_LAYOUT_COL_MAJOR));

        // Triangular solve: x = A^{-1} b
        CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_SOLVE, m_config, m_data,
                                 m_cudss_A, cudss_x, cudss_b));

        cudssMatrixDestroy(cudss_b);
        cudssMatrixDestroy(cudss_x);
    }

    // Mark solution as dirty for halo exchange
    x.dirtybit = 1;
    return AMGX_ST_CONVERGED;
#else
    FatalError("AMGx was not built with cuDSS support", AMGX_ERR_NOT_IMPLEMENTED);
    return AMGX_ST_ERROR;
#endif
}

// ============================================================
// solver_setup — symbolic analysis + numeric factorization
// ============================================================
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void CudssSolver<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>::solver_setup(
    bool reuse_matrix_structure)
{
#ifdef AMGX_USE_CUDSS
    Matrix_d *A = dynamic_cast<Matrix_d *>(Base::m_A);

    if (!A)
    {
        FatalError("CudssSolver only works with explicit matrices", AMGX_ERR_INTERNAL);
    }

    int block_dimx = A->get_block_dimx();
    int block_dimy = A->get_block_dimy();

    // Set the CUDA stream on the cuDSS handle to match AMGx's stream
    cudaStream_t stream = amgx::thrust::global_thread_handle::get_stream();
    CUDSS_CHECK(cudssSetStream(m_handle, stream));

    // Determine cuDSS matrix type from config
    cudssMatrixType_t mtype = CUDSS_MTYPE_GENERAL;
    cudssMatrixViewType_t mview = CUDSS_MVIEW_FULL;

    if (m_matrix_type == "SPD")
    {
        mtype = CUDSS_MTYPE_SPD;
        mview = CUDSS_MVIEW_UPPER;
    }
    else if (m_matrix_type == "SYMMETRIC")
    {
        mtype = CUDSS_MTYPE_SYMMETRIC;
        mview = CUDSS_MVIEW_UPPER;
    }

    // Configure reordering algorithm
    cudssAlgType_t reorder_alg;
    switch (m_reorder)
    {
        case 0:  reorder_alg = CUDSS_ALG_0; break;
        case 1:  reorder_alg = CUDSS_ALG_DEFAULT; break;
        case 2:  reorder_alg = CUDSS_ALG_1; break;
        default: reorder_alg = CUDSS_ALG_DEFAULT; break;
    }
    CUDSS_CHECK(cudssConfigSet(m_config, CUDSS_REORDERING_ALG,
                               &reorder_alg, sizeof(reorder_alg)));

    if (A->is_matrix_distributed() && m_enable_exact_solve)
    {
        // ============================================================
        // Distributed exact solve: gather global CSR on every rank,
        // then factorize the full global matrix with cuDSS.
        // Mirrors the Dense_LU_Solver exact coarse solve pattern.
        // ============================================================
#ifdef AMGX_WITH_MPI
        int rank   = A->manager->global_id();
        int nranks = A->manager->get_num_partitions();

        int offset, num_rows, nnz;
        A->getOffsetAndSizeForView(OWNED, &offset, &num_rows);
        A->getNnzForView(OWNED, &nnz);

        m_num_rows_global = A->manager->num_rows_global;

        if (!reuse_matrix_structure || !m_structure_set)
        {
            // Gather nnz and row counts from all ranks
            A->manager->getComms()->all_gather(nnz, m_nz_all, nranks);
            A->manager->getComms()->all_gather(num_rows, m_row_all, nranks);

            m_nnz_global = thrust_wrapper::reduce<AMGX_host>(
                m_nz_all.begin(), m_nz_all.end());

            // Compute displacements via exclusive scan
            m_nz_displs.resize(nranks);
            thrust_wrapper::exclusive_scan<AMGX_host>(
                m_nz_all.begin(), m_nz_all.end(), m_nz_displs.begin());

            m_row_displs.resize(nranks);
            thrust_wrapper::exclusive_scan<AMGX_host>(
                m_row_all.begin(), m_row_all.end(), m_row_displs.begin());

            // Copy local col_indices and row_offsets, convert local→global
            IVector_d local_Acols_d(nnz);
            IVector_d local_Arows_d(num_rows);

            amgx::thrust::copy(
                A->col_indices.begin(), A->col_indices.begin() + nnz,
                local_Acols_d.begin());
            amgx::thrust::copy(
                A->row_offsets.begin(), A->row_offsets.begin() + num_rows,
                local_Arows_d.begin());

            // Convert local column indices to global index space
            constexpr int nthreads = 128;
            int nblocks = (nnz + nthreads - 1) / nthreads;
            local_col_indices_to_global<<<nblocks, nthreads>>>(
                nnz, num_rows, m_row_displs[rank],
                local_Acols_d.raw(),
                A->manager->local_to_global_map.raw());
            cudaCheckError();

            // Convert local row offsets to global
            nblocks = (num_rows + nthreads - 1) / nthreads;
            local_row_offsets_to_global<<<nblocks, nthreads>>>(
                num_rows, m_nz_displs[rank], local_Arows_d.raw());
            cudaCheckError();

            // Copy transformed indices to host for all_gather_v
            IVector_h local_Acols_h(nnz);
            IVector_h local_Arows_h(num_rows);
            amgx::thrust::copy(
                local_Acols_d.begin(), local_Acols_d.end(),
                local_Acols_h.begin());
            amgx::thrust::copy(
                local_Arows_d.begin(), local_Arows_d.end(),
                local_Arows_h.begin());

            // Gather global matrix structure from all ranks
            IVector_h Acols_global_h(m_nnz_global);
            A->manager->getComms()->all_gather_v(
                local_Acols_h, nnz, Acols_global_h,
                m_nz_all, m_nz_displs);

            IVector_h Arows_global_h(m_num_rows_global + 1);
            A->manager->getComms()->all_gather_v(
                local_Arows_h, num_rows, Arows_global_h,
                m_row_all, m_row_displs);

            // Set the guard value for the global row offsets
            Arows_global_h[m_num_rows_global] = m_nnz_global;

            // Copy global structure to device
            m_Acols_global.resize(m_nnz_global);
            m_Arows_global.resize(m_num_rows_global + 1);
            amgx::thrust::copy(
                Acols_global_h.begin(), Acols_global_h.end(),
                m_Acols_global.begin());
            amgx::thrust::copy(
                Arows_global_h.begin(), Arows_global_h.end(),
                m_Arows_global.begin());
        }

        // Gather matrix values (must be done every setup, even for reuse)
        MVector_h local_Avals_h(nnz);
        amgx::thrust::copy(
            A->values.begin(), A->values.begin() + nnz,
            local_Avals_h.begin());

        MVector_h Avals_global_h(m_nnz_global);
        A->manager->getComms()->all_gather_v(
            local_Avals_h, nnz, Avals_global_h,
            m_nz_all, m_nz_displs);

        m_Avals_global.resize(m_nnz_global);
        amgx::thrust::copy(
            Avals_global_h.begin(), Avals_global_h.end(),
            m_Avals_global.begin());

        if (reuse_matrix_structure && m_structure_set)
        {
            // Reuse: update value pointers and refactorize
            CUDSS_CHECK(cudssMatrixSetCsrPointers(
                m_cudss_A,
                m_Arows_global.raw(), nullptr,
                m_Acols_global.raw(), m_Avals_global.raw()));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_REFACTORIZATION,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));
        }
        else
        {
            // Full setup: destroy old descriptor if it exists
            if (m_cudss_A != nullptr)
            {
                cudssMatrixDestroy(m_cudss_A);
                m_cudss_A = nullptr;
            }
            if (m_structure_set)
            {
                cudssDataDestroy(m_handle, m_data);
                CUDSS_CHECK(cudssDataCreate(m_handle, &m_data));
            }

            // Create CSR descriptor for the global gathered matrix
            CUDSS_CHECK(cudssMatrixCreateCsr(
                &m_cudss_A,
                static_cast<int64_t>(m_num_rows_global),
                static_cast<int64_t>(m_num_rows_global),
                static_cast<int64_t>(m_nnz_global),
                m_Arows_global.raw(), nullptr,
                m_Acols_global.raw(), m_Avals_global.raw(),
                CUDA_R_32I, cudss_data_type<ValueTypeA>(),
                mtype, mview, CUDSS_BASE_ZERO));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_ANALYSIS,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_FACTORIZATION,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));

            m_structure_set = true;
        }
#endif
    }
    else
    {
        // ============================================================
        // Single-GPU or inexact (local diagonal block) path
        // ============================================================
        int n_block = A->get_num_rows();
        int nnz_block = A->get_num_nz();

        if (n_block == 0)
        {
            m_factored = true;
            return;
        }

        // Determine whether we need to expand block-CSR to scalar CSR.
        // cuDSS requires scalar CSR; AMGx block-CSR stores bm×bn blocks.
        const bool is_block = (block_dimx > 1 || block_dimy > 1);
        const int  n_scalar   = n_block   * block_dimx;
        const int  nnz_scalar = nnz_block * block_dimx * block_dimy;

        // Pointers that will be passed to cuDSS (either raw AMGx arrays or
        // the expanded scalar arrays, depending on block_size).
        IndexType  *row_ptr = nullptr;
        IndexType  *col_ptr = nullptr;
        ValueTypeA *val_ptr = nullptr;

        if (is_block && (!reuse_matrix_structure || !m_structure_set))
        {
            // Expand block-CSR → scalar CSR on the device.
            m_scalar_row_offsets.resize(n_scalar + 1);
            m_scalar_col_indices.resize(nnz_scalar);
            m_scalar_values.resize(nnz_scalar);

            constexpr int nthreads = 128;
            int nblocks_rows = (n_scalar + 1 + nthreads - 1) / nthreads;

            // Phase 1: compute scalar row offsets
            block_csr_row_offsets_kernel<IndexType>
                <<<nblocks_rows, nthreads>>>(
                    n_block, block_dimx, block_dimy,
                    A->row_offsets.raw(),
                    m_scalar_row_offsets.raw());
            cudaCheckError();

            // Phase 2: fill scalar col_indices and values
            int nblocks_scalar = (n_scalar + nthreads - 1) / nthreads;
            block_csr_fill_cols_vals_kernel<IndexType, ValueTypeA>
                <<<nblocks_scalar, nthreads>>>(
                    n_block, block_dimx, block_dimy,
                    A->row_offsets.raw(),
                    A->col_indices.raw(),
                    A->values.raw(),
                    m_scalar_row_offsets.raw(),
                    m_scalar_col_indices.raw(),
                    m_scalar_values.raw());
            cudaCheckError();

            row_ptr = m_scalar_row_offsets.raw();
            col_ptr = m_scalar_col_indices.raw();
            val_ptr = m_scalar_values.raw();
        }
        else if (is_block && reuse_matrix_structure && m_structure_set)
        {
            // Reuse structure: only re-expand values (structure unchanged).
            // Re-run fill kernel to update values in m_scalar_values.
            constexpr int nthreads = 128;
            int nblocks_scalar = (n_scalar + nthreads - 1) / nthreads;
            block_csr_fill_cols_vals_kernel<IndexType, ValueTypeA>
                <<<nblocks_scalar, nthreads>>>(
                    n_block, block_dimx, block_dimy,
                    A->row_offsets.raw(),
                    A->col_indices.raw(),
                    A->values.raw(),
                    m_scalar_row_offsets.raw(),
                    m_scalar_col_indices.raw(),
                    m_scalar_values.raw());
            cudaCheckError();

            row_ptr = m_scalar_row_offsets.raw();
            col_ptr = m_scalar_col_indices.raw();
            val_ptr = m_scalar_values.raw();
        }
        else
        {
            // Scalar CSR (block_size == 1): use AMGx arrays directly.
            row_ptr = A->row_offsets.raw();
            col_ptr = A->col_indices.raw();
            val_ptr = A->values.raw();
        }

        const int64_t n_csr   = static_cast<int64_t>(n_scalar);
        const int64_t nnz_csr = static_cast<int64_t>(is_block ? nnz_scalar : nnz_block);

        if (reuse_matrix_structure && m_structure_set)
        {
            CUDSS_CHECK(cudssMatrixSetCsrPointers(
                m_cudss_A,
                row_ptr, nullptr,
                col_ptr, val_ptr));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_REFACTORIZATION,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));
        }
        else
        {
            if (m_cudss_A != nullptr)
            {
                cudssMatrixDestroy(m_cudss_A);
                m_cudss_A = nullptr;
            }
            if (m_structure_set)
            {
                cudssDataDestroy(m_handle, m_data);
                CUDSS_CHECK(cudssDataCreate(m_handle, &m_data));
            }

            CUDSS_CHECK(cudssMatrixCreateCsr(
                &m_cudss_A,
                n_csr, n_csr, nnz_csr,
                row_ptr, nullptr,
                col_ptr, val_ptr,
                CUDA_R_32I, cudss_data_type<ValueTypeA>(),
                mtype, mview, CUDSS_BASE_ZERO));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_ANALYSIS,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));

            CUDSS_CHECK(cudssExecute(m_handle, CUDSS_PHASE_FACTORIZATION,
                                     m_config, m_data, m_cudss_A,
                                     nullptr, nullptr));

            m_structure_set = true;
        }
    }

    m_factored = true;
#else
    FatalError("AMGx was not built with cuDSS support", AMGX_ERR_NOT_IMPLEMENTED);
#endif
}

// ============================================================
// Template instantiation
// ============================================================
#define AMGX_CASE_LINE(CASE) template class CudssSolver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace cudss_solver
} // namespace amgx
