// SPDX-FileCopyrightText: 2024 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

// MIS-k (Maximal Independent Set, distance k) aggregation selector for SA-AMG.
//
// Algorithm overview:
//   Phase 1 — Compute MIS-k:
//     Assign deterministic hash-based weights to each node.
//     Iteratively mark nodes as MIS_ROOT (highest weight among undecided
//     distance-k neighbors) or NOT_IN_MIS (has a MIS_ROOT within distance k).
//   Phase 2 — Assign non-MIS nodes to nearest root aggregate.
//   Phase 3 — Propagate assignments to any remaining unassigned nodes.
//   Phase 4 — Renumber aggregates contiguously via base-class helper.
//
// This produces roughly spherical aggregates with bounded diameter, which
// is critical for SA-AMG convergence quality (h-independent convergence).
//
// MIS-k for k>1 is implemented via a Galerkin coarsening loop:
//   R_out_agg[i] = i  (identity)
//   for pass = 0 to k-1:
//       Run MIS-1 on A_cur (with exchange_halo for MPI correctness — Step 1)
//       Compose R_out_agg via thrust::gather
//       If not last pass: A_cur = R * A_cur * R^T  (Galerkin product)
//   Output: aggregates[] = R_out_agg[]

#include <aggregation/selectors/mis_selector.h>
#include <cutil.h>
#include <util.h>
#include <types.h>
#include <basic_types.h>
#include <matrix_analysis.h>
#include <csr_multiply.h>
#include <transpose.h>
#include <texture.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <cusp/detail/format_utils.h>  // offsets_to_indices

namespace amgx
{
namespace aggregation
{
namespace mis_selector
{

// Include common routines (hash_val, computeEdgeWeightsBlockDiaCsr_V2, etc.)
#include <aggregation/selectors/common_selector.h>

// -----------------------------------------------------------------------
// Node status constants for MIS computation
// -----------------------------------------------------------------------
static const int MIS_UNDECIDED  = -1;
static const int MIS_ROOT       =  1;
static const int MIS_NOT_ROOT   =  0;

// -----------------------------------------------------------------------
// Named device functors to replace __device__ lambdas.
// CUDA does not allow extended __device__ lambdas inside private or
// protected member functions, so we use named functors at namespace scope.
// -----------------------------------------------------------------------

// Functor: set any value to MIS_ROOT (used in transform_if)
struct SetMISRoot
{
    __device__ int operator()(int /*s*/) const { return MIS_ROOT; }
};

// Functor: predicate — true if status is MIS_UNDECIDED
struct IsUndecided
{
    __device__ bool operator()(int s) const { return s == MIS_UNDECIDED; }
};

// Functor: cast float edge weight to ValueType (template)
template <typename ValueType>
struct FloatToValueType
{
    __device__ ValueType operator()(float v) const { return (ValueType)v; }
};

// Functor: absolute value of ValueType cast to float (for coarse edge weights)
template <typename ValueType>
struct AbsToFloat
{
    __device__ float operator()(ValueType v) const { return fabsf((float)v); }
};

// -----------------------------------------------------------------------
// Kernel: assign_node_weights
//
// Assigns a deterministic pseudo-random weight in [0,1) to each node
// using the hash_val() function from common_selector.h.
// An optional seed offset allows different weights per Galerkin pass.
// -----------------------------------------------------------------------
__global__
void assign_node_weights_kernel(const int num_rows, float *node_weights,
                                unsigned int seed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        unsigned int h = hash_val((unsigned int)tid, seed);
        node_weights[tid] = (float)h / (float)UINT_MAX;
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: find_mis_k1_kernel  (MIS distance k=1)
//
// Each undecided node checks its direct neighbors:
//   - If any neighbor is already MIS_ROOT → mark self NOT_IN_MIS
//   - Else if self has the highest weight among all undecided neighbors
//     (and no neighbor is MIS_ROOT) → mark self MIS_ROOT
//
// num_rows   = number of owned rows (writes only to tid < num_rows)
// total_rows = owned + halo rows (valid status/weight indices up to total_rows)
//
// After Step 1 (exchange_halo), halo entries in status[] and node_weights[]
// are valid, so we use total_rows as the upper bound for neighbor reads.
//
// Returns the number of nodes that changed status this round via
// atomicAdd into *num_changed.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void find_mis_k1_kernel(const IndexType *row_offsets,
                        const IndexType *col_indices,
                        const float     *node_weights,
                        int             *status,
                        const int        num_rows,
                        const int        total_rows,
                        int             *num_changed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (status[tid] == MIS_UNDECIDED)
        {
            float my_weight = node_weights[tid];
            bool dominated  = false;   // a MIS_ROOT neighbor exists
            bool is_max     = true;    // highest weight among undecided neighbors

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                int jcol = col_indices[j];
                // Skip self-loops; skip columns beyond valid range (total_rows
                // covers owned + halo after exchange_halo; on single-GPU
                // total_rows == num_rows so this is equivalent to the old guard)
                if (jcol == tid || jcol >= total_rows) { continue; }

                int jstatus = status[jcol];
                if (jstatus == MIS_ROOT)
                {
                    dominated = true;
                    break;
                }
                // Among undecided neighbors, check if we have the max weight.
                // Tie-break by index (higher index wins) for determinism.
                if (jstatus == MIS_UNDECIDED)
                {
                    float jw = node_weights[jcol];
                    if (jw > my_weight || (jw == my_weight && jcol > tid))
                    {
                        is_max = false;
                    }
                }
            }

            if (dominated)
            {
                status[tid] = MIS_NOT_ROOT;
                atomicAdd(num_changed, 1);
            }
            else if (is_max)
            {
                status[tid] = MIS_ROOT;
                atomicAdd(num_changed, 1);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: find_mis_k2_kernel  (MIS distance k=2, legacy single-pass path)
//
// Same as k=1 but also checks distance-2 neighbors (neighbors of neighbors).
// A node is dominated if any distance-1 or distance-2 neighbor is MIS_ROOT.
// A node is the local maximum if it has the highest weight among all
// undecided nodes within distance 2.
//
// NOTE: This kernel is retained for reference but is superseded by the
// Galerkin loop (which uses find_mis_k1_kernel exclusively).
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void find_mis_k2_kernel(const IndexType *row_offsets,
                        const IndexType *col_indices,
                        const float     *node_weights,
                        int             *status,
                        const int        num_rows,
                        int             *num_changed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (status[tid] == MIS_UNDECIDED)
        {
            float my_weight = node_weights[tid];
            bool dominated  = false;
            bool is_max     = true;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            // Check distance-1 neighbors
            for (int j = jmin; j < jmax && !dominated; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) { continue; }

                int jstatus = status[jcol];
                if (jstatus == MIS_ROOT)
                {
                    dominated = true;
                    break;
                }
                if (jstatus == MIS_UNDECIDED)
                {
                    float jw = node_weights[jcol];
                    if (jw > my_weight || (jw == my_weight && jcol > tid))
                    {
                        is_max = false;
                    }
                }

                // Check distance-2 neighbors (neighbors of jcol)
                int kmin = row_offsets[jcol];
                int kmax = row_offsets[jcol + 1];
                for (int k = kmin; k < kmax && !dominated; k++)
                {
                    int kcol = col_indices[k];
                    if (kcol == tid || kcol == jcol || kcol >= num_rows) { continue; }

                    int kstatus = status[kcol];
                    if (kstatus == MIS_ROOT)
                    {
                        dominated = true;
                        break;
                    }
                    if (kstatus == MIS_UNDECIDED && is_max)
                    {
                        float kw = node_weights[kcol];
                        if (kw > my_weight || (kw == my_weight && kcol > tid))
                        {
                            is_max = false;
                        }
                    }
                }
            }

            if (dominated)
            {
                status[tid] = MIS_NOT_ROOT;
                atomicAdd(num_changed, 1);
            }
            else if (is_max)
            {
                status[tid] = MIS_ROOT;
                atomicAdd(num_changed, 1);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: assign_aggregates_kernel
//
// For each MIS_ROOT node: aggregates[i] = i  (it is its own aggregate root)
// For each MIS_NOT_ROOT node: find the strongest-connected MIS_ROOT neighbor
// and assign aggregates[i] = that root's index.
// Nodes with no MIS_ROOT neighbor remain at -1 (handled by propagation).
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void assign_aggregates_kernel(const IndexType *row_offsets,
                              const IndexType *col_indices,
                              const float     *edge_weights,
                              const int       *status,
                              IndexType       *aggregates,
                              const int        num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (status[tid] == MIS_ROOT)
        {
            aggregates[tid] = tid;
        }
        else // MIS_NOT_ROOT — find strongest MIS_ROOT neighbor
        {
            float best_weight = -1.0f;
            int   best_root   = -1;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) { continue; }

                if (status[jcol] == MIS_ROOT)
                {
                    float w = edge_weights[j];
                    if (w > best_weight || (w == best_weight && jcol > best_root))
                    {
                        best_weight = w;
                        best_root   = jcol;
                    }
                }
            }

            aggregates[tid] = best_root; // may be -1 if no MIS_ROOT neighbor
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: propagate_aggregates_kernel
//
// For nodes still unassigned (aggregates[i] == -1), look for any assigned
// neighbor and adopt its aggregate.  Run iteratively until all assigned.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void propagate_aggregates_kernel(const IndexType *row_offsets,
                                 const IndexType *col_indices,
                                 const float     *edge_weights,
                                 IndexType       *aggregates,
                                 IndexType       *aggregates_candidate,
                                 const int        num_rows,
                                 const int        deterministic)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (aggregates[tid] == -1)
        {
            float best_weight = -1.0f;
            int   best_agg    = -1;
            int   best_col    = -1;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) { continue; }

                if (aggregates[jcol] != -1)
                {
                    float w = edge_weights[j];
                    if (w > best_weight || (w == best_weight && jcol > best_col))
                    {
                        best_weight = w;
                        best_agg    = aggregates[jcol];
                        best_col    = jcol;
                    }
                }
            }

            if (best_agg != -1)
            {
                if (deterministic)
                {
                    aggregates_candidate[tid] = best_agg;
                }
                else
                {
                    aggregates[tid] = best_agg;
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: join_candidates_kernel
//
// Apply the candidate assignments computed in deterministic propagation.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void join_candidates_kernel(IndexType *aggregates,
                            IndexType *aggregates_candidate,
                            const int  num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (aggregates[tid] == -1 && aggregates_candidate[tid] != -1)
        {
            aggregates[tid] = aggregates_candidate[tid];
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: make_singletons_kernel
//
// Any node still unassigned after propagation becomes its own singleton.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void make_singletons_kernel(IndexType *aggregates, const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        if (aggregates[tid] == -1)
        {
            aggregates[tid] = tid;
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: count_aggregate_sizes_kernel
//
// Counts how many nodes belong to each aggregate via atomicAdd.
// agg_sizes must be pre-zeroed before launch.
// -----------------------------------------------------------------------
__global__
void count_aggregate_sizes_kernel(const int *aggregates, int *agg_sizes,
                                   const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        int agg = aggregates[tid];
        if (agg >= 0)
        {
            atomicAdd(&agg_sizes[agg], 1);
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: compute_max_edge_weight_kernel
//
// For each row, find the maximum edge weight among its off-diagonal entries.
// Used to compute the per-row weak-edge threshold for quality refinement.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void compute_max_edge_weight_kernel(const IndexType *row_offsets,
                                     const IndexType *col_indices,
                                     const float *edge_weights,
                                     float *max_edge_weight,
                                     const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        float max_w = 0.0f;
        int jmin = row_offsets[tid];
        int jmax = row_offsets[tid + 1];
        for (int j = jmin; j < jmax; j++)
        {
            int jcol = col_indices[j];
            if (jcol == tid) continue;  // skip diagonal
            float w = edge_weights[j];
            if (w > max_w) max_w = w;
        }
        max_edge_weight[tid] = max_w;
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: refine_aggregates_kernel (REVISED)
//
// For nodes in oversized aggregates (size > max_agg_size):
//   1. Find the best neighbor in a SMALLER aggregate (any size, not just < max)
//      using score = edge_weight / target_agg_size
//   2. If no smaller neighbor found, become a new singleton (aggregates[tid] = tid)
//      BUT only if the node has at least one neighbor in a different aggregate
//      (to avoid fragmenting connected components)
//
// The "become singleton" behavior creates new aggregates that will attract
// nearby nodes in subsequent iterations, effectively splitting large aggregates.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void refine_aggregates_kernel(
    const IndexType *row_offsets,
    const IndexType *col_indices,
    const float     *edge_weights,
    const float     *max_edge_weight,
    IndexType       *aggregates,
    const int       *agg_sizes,
    const int        max_agg_size,
    const float      alpha,
    const int        num_rows,
    int             *num_reassigned)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        int my_agg = aggregates[tid];
        // Only process nodes in oversized aggregates
        if (my_agg >= 0 && agg_sizes[my_agg] > max_agg_size)
        {
            float threshold = max_edge_weight[tid] * alpha;
            float best_score = -1.0f;
            int   best_agg   = -1;
            bool  has_diff_neighbor = false;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) continue;

                float w = edge_weights[j];
                if (w < threshold) continue;  // skip weak edges

                int jagg = aggregates[jcol];
                if (jagg >= 0 && jagg != my_agg)
                {
                    has_diff_neighbor = true;
                    // Move to any SMALLER aggregate (not just < max_agg_size)
                    if (agg_sizes[jagg] < agg_sizes[my_agg])
                    {
                        float score = w / (float)(agg_sizes[jagg] + 1);
                        if (score > best_score)
                        {
                            best_score = score;
                            best_agg = jagg;
                        }
                    }
                }
            }

            if (best_agg >= 0)
            {
                // Move to a smaller neighbor aggregate
                aggregates[tid] = best_agg;
                atomicAdd(num_reassigned, 1);
            }
            else if (has_diff_neighbor)
            {
                // No smaller neighbor found, but we're at an aggregate boundary
                // Become a new singleton to seed a new aggregate
                aggregates[tid] = tid;
                atomicAdd(num_reassigned, 1);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Constructor
// -----------------------------------------------------------------------
template<class T_Config>
MISSelectorBase<T_Config>::MISSelectorBase(AMG_Config &cfg, const std::string &cfg_scope)
{
    m_mis_k = cfg.AMG_Config::template getParameter<int>("mis_k", cfg_scope);
    m_aggressive_levels = cfg.AMG_Config::template getParameter<int>("aggressive_levels", cfg_scope);
    m_max_iterations = cfg.AMG_Config::template getParameter<int>("max_matching_iterations", cfg_scope);
    m_merge_singletons = cfg.AMG_Config::template getParameter<int>("merge_singletons", cfg_scope);
    m_weight_formula = cfg.AMG_Config::template getParameter<int>("weight_formula", cfg_scope);
    m_aggregation_edge_weight_component = cfg.AMG_Config::template getParameter<int>("aggregation_edge_weight_component", cfg_scope);
    m_call_count = 0;
    m_max_aggregate_size = cfg.AMG_Config::template getParameter<int>("max_aggregate_size", cfg_scope);
    m_refine_threshold = cfg.AMG_Config::template getParameter<double>("refine_threshold", cfg_scope);
    m_mis2_algorithm = cfg.AMG_Config::template getParameter<int>("mis2_algorithm", cfg_scope);
}

// -----------------------------------------------------------------------
// setAggregates — dispatch to block-size-specific implementation
// -----------------------------------------------------------------------
template<class T_Config>
void MISSelectorBase<T_Config>::setAggregates(Matrix<T_Config> &A,
        IVector &aggregates, IVector &aggregates_global, int &num_aggregates)
{
    if (A.get_block_dimx() == A.get_block_dimy())
    {
        setAggregates_common_sqblocks(A, aggregates, aggregates_global, num_aggregates);
    }
    else
    {
        FatalError("MIS selector: unsupported non-square block size", AMGX_ERR_NOT_SUPPORTED_BLOCKSIZE);
    }
}

// -----------------------------------------------------------------------
// Host specialization — not implemented (GPU-only selector)
// -----------------------------------------------------------------------
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MISSelector<TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(
    const Matrix_h &A,
    typename Matrix_h::IVector &aggregates,
    typename Matrix_h::IVector &aggregates_global,
    int &num_aggregates)
{
    FatalError("MIS selector: setAggregates not implemented on CPU", AMGX_ERR_NOT_SUPPORTED_TARGET);
}

// -----------------------------------------------------------------------
// Device specialization — main GPU implementation
//
// Implements the Galerkin coarsening loop for MIS-k:
//
//   R_out_agg[i] = i  (identity)
//   for pass = 0 to effective_k-1:
//       Compute edge weights on A_cur
//       Run MIS-1 with exchange_halo (Step 1 MPI fix)
//       Assign all nodes to aggregates
//       R_out_agg[i] = agg_cur[R_out_agg[i]]  (compose via thrust::gather)
//       If not last pass: A_cur = R * A_cur * R^T  (Galerkin product)
//   Output aggregates[] = renumber(R_out_agg)
// -----------------------------------------------------------------------
template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void MISSelector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> >::setAggregates_common_sqblocks(
    const Matrix_d &A,
    typename Matrix_d::IVector &aggregates,
    typename Matrix_d::IVector &aggregates_global,
    int &num_aggregates)
{
    typedef TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec> TConfig_d;
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecFloat>::Type> FVector_d;
    typedef Vector<typename TConfig_d::template setVecPrec<AMGX_vecInt>::Type>   IVector_d;

    const IndexType num_block_rows  = (IndexType) A.get_num_rows();
    const IndexType num_nonzero_blk = (IndexType) A.get_num_nz();

    // Resize aggregates to cover owned + halo rows
    IndexType total_rows = A.is_matrix_singleGPU() ? num_block_rows
                                                    : A.manager->num_rows_all();
    aggregates.resize(total_rows);
    thrust_wrapper::fill<AMGX_device>(aggregates.begin(), aggregates.end(), -1);
    cudaCheckError();

    cudaStream_t str = amgx::thrust::global_thread_handle::get_stream();

    const int threads_per_block = 256;

    // ------------------------------------------------------------------
    // Multi-GPU fallback for mis_k > 1
    // TODO (Step 4): Multi-GPU MIS-k support
    // Use setNeighborAggregates() to set up Ac.manager from A.manager + agg_cur,
    // then computeAOperator() for the distributed Galerkin product,
    // then prepareNextLevelMatrix() to finalize halo rows.
    // These functions are in aggregation_amg_level.cu and handle all
    // distributed manager setup (B2L_maps, halo_offsets, renumbering).
    // ------------------------------------------------------------------
    int effective_k = this->m_mis_k;

    // Per-level aggressive coarsening control:
    // aggressive_levels=0 means use mis_k on ALL levels
    // aggressive_levels=N means use mis_k on first N levels, mis_k=1 on rest
    // Read level index from matrix parameter (set by aggregation_amg_level.cu)
    int current_level = 0;
    try { current_level = A.template getParameter<int>("amg_level_index"); }
    catch (...) { current_level = 0; }
    if (this->m_aggressive_levels > 0 && current_level >= this->m_aggressive_levels)
    {
        effective_k = 1;  // fall back to MIS-1 on non-aggressive levels
    }

    if (!A.is_matrix_singleGPU() && effective_k > 1)
    {
        amgx_printf("WARNING: mis_k=%d not yet supported on multi-GPU, "
                    "falling back to mis_k=1\n", effective_k);
        effective_k = 1;
    }

    amgx_printf("[MIS-k] Level %d: mis_k=%d, effective_k=%d "
                "(aggressive_levels=%d)\n",
                current_level, this->m_mis_k, effective_k,
                this->m_aggressive_levels);

    // ------------------------------------------------------------------
    // Initialize composed aggregation: R_out_agg[i] = i (identity)
    // ------------------------------------------------------------------
    IVector_d R_out_agg(num_block_rows);
    thrust::sequence(thrust::cuda::par.on(str),
                     R_out_agg.begin(), R_out_agg.end());
    cudaCheckError();

    // ------------------------------------------------------------------
    // Build CSR row-index array for edge-weight kernel (pass 0 only)
    // ------------------------------------------------------------------
    IndexType total_nz_orig = A.is_matrix_singleGPU() ? num_nonzero_blk
                                                       : A.manager->num_nz_all();
    typename Matrix_d::IVector row_indices_orig(total_nz_orig);
    cusp::detail::offsets_to_indices(A.row_offsets, row_indices_orig);

    const int num_blocks_nz_orig = std::min(AMGX_GRID_MAX_SIZE,
                                            (num_nonzero_blk - 1) / threads_per_block + 1);

    // ------------------------------------------------------------------
    // Compute edge weights for pass 0 (block matrix)
    // ------------------------------------------------------------------
    FVector_d edge_weights_orig(num_nonzero_blk, 0.0f);

    cudaFuncSetCacheConfig(
        computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, float>,
        cudaFuncCachePreferL1);
    computeEdgeWeightsBlockDiaCsr_V2<IndexType, ValueType, float>
        <<<num_blocks_nz_orig, threads_per_block, 0, str>>>(
        A.row_offsets.raw(), (const IndexType *)row_indices_orig.raw(), A.col_indices.raw(),
        A.diag.raw(), A.values.raw(),
        num_nonzero_blk, edge_weights_orig.raw(), /*rand_edge_weights=*/nullptr,
        num_block_rows, A.get_block_dimy(),
        this->m_aggregation_edge_weight_component,
        this->m_weight_formula);
    cudaCheckError();

    // ------------------------------------------------------------------
    // Galerkin coarsening loop
    //
    // A_cur_ptr       — pointer to current matrix (A on pass 0, A_coarse on pass 1+)
    // edge_weights_cur_ptr — pointer to current edge weights
    // A_coarse        — storage for Galerkin coarse matrix (pass 1+)
    // edge_weights_coarse — storage for coarse edge weights (pass 1+)
    // ------------------------------------------------------------------
    const Matrix_d *A_cur_ptr            = &A;
    Matrix_d        A_coarse;
    FVector_d      *edge_weights_cur_ptr = &edge_weights_orig;
    FVector_d       edge_weights_coarse;

    for (int pass = 0; pass < effective_k; pass++)
    {
        const Matrix_d &A_cur = *A_cur_ptr;
        int n_cur     = A_cur.get_num_rows();
        int nnz_cur_total = A_cur.get_num_nz();
        int total_cur = A_cur.is_matrix_singleGPU() ? n_cur
                                                     : A_cur.manager->num_rows_all();

        const int num_blocks_cur = std::min(AMGX_GRID_MAX_SIZE,
                                            (n_cur - 1) / threads_per_block + 1);

        if (effective_k > 1)
        {
            amgx_printf("[MIS-k] Pass %d/%d: n_cur=%d, nnz=%d, avg_nnz/row=%.1f\n",
                        pass, effective_k, n_cur, nnz_cur_total,
                        (float)nnz_cur_total / (float)n_cur);
        }

        // ----------------------------------------------------------------
        // Phase 1: Assign node weights and run MIS-1 with exchange_halo
        //
        // node_weights and status_vec are sized to total_cur (owned + halo)
        // so that exchange_halo can fill halo entries in-place.
        // ----------------------------------------------------------------

        // Use a different hash seed per pass to decorrelate weights
        unsigned int seed = 0x9e3779b9u + (unsigned int)(pass * 0x6c62272eu);

        // Allocate to total_cur so exchange_halo can write halo entries
        FVector_d node_weights(total_cur, 0.0f);
        assign_node_weights_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
            n_cur, node_weights.raw(), seed);
        cudaCheckError();

        // Exchange node_weights halo so boundary comparisons use correct
        // remote weights (Step 1 MPI fix: also applies to node_weights)
        if (!A_cur.is_matrix_singleGPU())
        {
            node_weights.dirtybit = 1;
            A_cur.manager->exchange_halo(node_weights, node_weights.tag);
        }

        // status_vec sized to total_cur; halo entries start as MIS_UNDECIDED
        // and are filled by exchange_halo after each iteration
        IVector_d status_vec(total_cur, MIS_UNDECIDED);
        int *status_ptr = status_vec.raw();

        IVector_d d_changed(1);
        int *d_changed_ptr = d_changed.raw();

        int num_undecided = n_cur;
        int icount = 0;

        while (num_undecided > 0 && icount < this->m_max_iterations)
        {
            cudaMemsetAsync(d_changed_ptr, 0, sizeof(int), str);

            // Always use find_mis_k1_kernel in the Galerkin loop;
            // the k-distance is achieved by iterating the loop effective_k times.
            find_mis_k1_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                A_cur.row_offsets.raw(), A_cur.col_indices.raw(),
                node_weights.raw(), status_ptr,
                n_cur, total_cur, d_changed_ptr);
            cudaCheckError();

            // Step 1 (MPI fix): exchange status so halo nodes reflect
            // remote decisions before the next iteration
            if (!A_cur.is_matrix_singleGPU())
            {
                status_vec.dirtybit = 1;
                A_cur.manager->exchange_halo(status_vec, status_vec.tag);
            }

            // Count remaining undecided owned nodes
            num_undecided = (int)thrust_wrapper::count<AMGX_device>(
                status_vec.begin(), status_vec.begin() + n_cur,
                MIS_UNDECIDED);
            cudaCheckError();
            icount++;
        }

        // Safety: any remaining undecided nodes become MIS roots
        if (num_undecided > 0)
        {
            thrust::transform_if(
                thrust::cuda::par.on(str),
                status_vec.begin(), status_vec.begin() + n_cur,
                status_vec.begin(),
                SetMISRoot(),
                IsUndecided());
            cudaCheckError();
        }

        // Diagnostic: count MIS roots for this pass
        if (effective_k > 1)
        {
            int n_roots_mis = (int)thrust_wrapper::count<AMGX_device>(
                status_vec.begin(), status_vec.begin() + n_cur, MIS_ROOT);
            cudaCheckError();
            amgx_printf("[MIS-k] Pass %d: MIS-1 converged in %d iters, "
                        "%d roots out of %d nodes (%.1f%%), "
                        "%d forced-root\n",
                        pass, icount, n_roots_mis, n_cur,
                        100.0f * n_roots_mis / n_cur,
                        num_undecided);
        }

        // ----------------------------------------------------------------
        // Phase 2: Assign non-MIS nodes to nearest MIS root
        // ----------------------------------------------------------------
        IVector_d agg_cur(n_cur, -1);
        assign_aggregates_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
            A_cur.row_offsets.raw(), A_cur.col_indices.raw(),
            edge_weights_cur_ptr->raw(), status_ptr,
            agg_cur.raw(), n_cur);
        cudaCheckError();

        // ----------------------------------------------------------------
        // Phase 3: Propagate assignments to nodes not adjacent to any root
        // ----------------------------------------------------------------
        int num_unassigned = (int)thrust_wrapper::count<AMGX_device>(
            agg_cur.begin(), agg_cur.begin() + n_cur, -1);
        cudaCheckError();

        if (num_unassigned > 0)
        {
            if (!this->m_merge_singletons)
            {
                make_singletons_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                    agg_cur.raw(), n_cur);
                cudaCheckError();
            }
            else
            {
                IVector_d agg_candidate(n_cur, -1);

                while (num_unassigned > 0)
                {
                    thrust_wrapper::fill<AMGX_device>(
                        agg_candidate.begin(), agg_candidate.end(), -1);

                    propagate_aggregates_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                        A_cur.row_offsets.raw(), A_cur.col_indices.raw(),
                        edge_weights_cur_ptr->raw(), agg_cur.raw(),
                        agg_candidate.raw(), n_cur, /*deterministic=*/1);
                    cudaCheckError();

                    join_candidates_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                        agg_cur.raw(), agg_candidate.raw(), n_cur);
                    cudaCheckError();

                    int prev_unassigned = num_unassigned;
                    num_unassigned = (int)thrust_wrapper::count<AMGX_device>(
                        agg_cur.begin(), agg_cur.begin() + n_cur, -1);
                    cudaCheckError();

                    // Guard against infinite loop (disconnected graph components)
                    if (num_unassigned == prev_unassigned)
                    {
                        make_singletons_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                            agg_cur.raw(), n_cur);
                        cudaCheckError();
                        break;
                    }
                }
            }
        }

        // ----------------------------------------------------------------
        // Phase 4: Renumber agg_cur to contiguous 0..n_roots-1
        // (required so Galerkin product dimensions are correct)
        // ----------------------------------------------------------------
        IVector_d agg_global_cur;
        int n_roots = 0;
        this->renumberAndCountAggregates(agg_cur, agg_global_cur, n_cur, n_roots);

        if (effective_k > 1)
        {
            amgx_printf("[MIS-k] Pass %d: %d aggregates from %d nodes "
                        "(coarsening ratio %.2fx)\n",
                        pass, n_roots, n_cur,
                        (float)n_cur / (float)n_roots);
        }

        // ----------------------------------------------------------------
        // Phase 5: Compose R_out_agg
        // R_out_agg[i] = agg_cur[R_out_agg[i]]
        // ----------------------------------------------------------------
        {
            IVector_d tmp(num_block_rows);
            thrust::gather(thrust::cuda::par.on(str),
                           R_out_agg.begin(), R_out_agg.end(),
                           agg_cur.begin(), tmp.begin());
            cudaCheckError();
            R_out_agg.swap(tmp);
        }

        // ----------------------------------------------------------------
        // Phase 6: Galerkin product (skip on last pass)
        //
        // Build scalar A_scalar from edge_weights (block_size=1 required
        // by csr_galerkin_product), then compute A_coarse = R * A_scalar * P.
        // ----------------------------------------------------------------
        if (pass < effective_k - 1)
        {
            int nnz_cur = A_cur.get_num_nz();

            // Build A_scalar: same sparsity as A_cur, values = edge_weights
            Matrix_d A_scalar;
            A_scalar.set_initialized(0);
            A_scalar.addProps(CSR);
            A_scalar.delProps(COO);
            A_scalar.delProps(DIAG);
            A_scalar.setColsReorderedByColor(false);
            A_scalar.row_offsets  = A_cur.row_offsets;
            A_scalar.col_indices  = A_cur.col_indices;
            A_scalar.values.resize(nnz_cur);
            thrust::transform(
                thrust::cuda::par.on(str),
                edge_weights_cur_ptr->begin(),
                edge_weights_cur_ptr->begin() + nnz_cur,
                A_scalar.values.begin(),
                FloatToValueType<ValueType>());
            cudaCheckError();
            A_scalar.resize(n_cur, n_cur, nnz_cur, 1, 1, false);
            A_scalar.set_initialized(1);

            // Build P: n_cur × n_roots, one nonzero per row at agg_cur[i]
            Matrix_d P_cur;
            P_cur.set_initialized(0);
            P_cur.addProps(CSR);
            P_cur.delProps(COO);
            P_cur.delProps(DIAG);
            P_cur.setColsReorderedByColor(false);
            P_cur.row_offsets.resize(n_cur + 1);
            P_cur.values.resize(n_cur, types::util<ValueType>::get_one());
            P_cur.col_indices.resize(n_cur);
            thrust_wrapper::sequence<AMGX_device>(
                P_cur.row_offsets.begin(), P_cur.row_offsets.end());
            amgx::thrust::copy(agg_cur.begin(), agg_cur.end(),
                               P_cur.col_indices.begin());
            cudaCheckError();
            P_cur.resize(n_cur, n_roots, n_cur, 1, 1, false);
            P_cur.set_initialized(1);

            // R = P^T  (n_roots × n_cur)
            Matrix_d R_cur;
            R_cur.set_initialized(0);
            R_cur.addProps(CSR);
            R_cur.delProps(COO);
            R_cur.delProps(DIAG);
            R_cur.setColsReorderedByColor(false);
            transpose(P_cur, R_cur);

            // A_coarse = R * A_scalar * P  (n_roots × n_roots)
            A_coarse.set_initialized(0);
            CSR_Multiply<TConfig_d>::csr_galerkin_product(
                R_cur, A_scalar, P_cur, A_coarse,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            A_coarse.computeDiagonal();
            A_coarse.set_initialized(1);

            // Compute edge weights for the coarse matrix: |values|
            int nnz_coarse = A_coarse.get_num_nz();
            edge_weights_coarse.resize(nnz_coarse);
            thrust::transform(
                thrust::cuda::par.on(str),
                A_coarse.values.begin(),
                A_coarse.values.begin() + nnz_coarse,
                edge_weights_coarse.begin(),
                AbsToFloat<ValueType>());
            cudaCheckError();

            A_cur_ptr            = &A_coarse;
            edge_weights_cur_ptr = &edge_weights_coarse;

            if (effective_k > 1)
            {
                int n_coarse = A_coarse.get_num_rows();
                amgx_printf("[MIS-k] Pass %d: Galerkin coarse matrix: "
                            "%d rows, %d nnz, avg_nnz/row=%.1f\n",
                            pass, n_coarse, nnz_coarse,
                            (float)nnz_coarse / (float)n_coarse);
            }
        }
    } // end Galerkin loop

    // ------------------------------------------------------------------
    // Final: copy composed aggregates to output and renumber
    // ------------------------------------------------------------------
    // aggregates was pre-sized to total_rows; fill owned portion from R_out_agg
    amgx::thrust::copy(R_out_agg.begin(), R_out_agg.end(), aggregates.begin());
    cudaCheckError();

    // Final renumber to produce contiguous 0..num_aggregates-1
    this->renumberAndCountAggregates(aggregates, aggregates_global,
                                     num_block_rows, num_aggregates);

    // ------------------------------------------------------------------
    // Quality-aware aggregate refinement
    //
    // After MIS-k produces initial aggregates, iteratively reassign nodes
    // from oversized aggregates to smaller neighbors using a quality score
    // that rewards strong edges and small target aggregates.
    //
    // Safe for asymmetric matrices: edge weights are symmetric by formula,
    // structural asymmetry results in weight=0 (ignored), per-row threshold
    // asymmetry is acceptable (unilateral decisions, no acceptance needed).
    //
    // MPI note: Currently only considers owned nodes (jcol < num_rows).
    // For MPI, edge weights for halo columns are 0 (known limitation).
    // ------------------------------------------------------------------
    if (this->m_max_aggregate_size > 0 && effective_k > 1)
    {
        const float alpha = (float)this->m_refine_threshold;
        const int max_refine_iters = 10;
        const int num_blocks_fine = std::min(AMGX_GRID_MAX_SIZE,
                                             (num_block_rows - 1) / threads_per_block + 1);

        // Compute max edge weight per row (for threshold)
        FVector_d max_ew_per_row(num_block_rows);
        compute_max_edge_weight_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
            A.row_offsets.raw(), A.col_indices.raw(),
            edge_weights_orig.raw(), max_ew_per_row.raw(), num_block_rows);
        cudaCheckError();

        IVector_d agg_sizes(num_aggregates);
        IVector_d d_reassigned(1);

        for (int refine_iter = 0; refine_iter < max_refine_iters; refine_iter++)
        {
            // Count aggregate sizes
            thrust_wrapper::fill<AMGX_device>(
                agg_sizes.begin(), agg_sizes.end(), 0);
            count_aggregate_sizes_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                aggregates.raw(), agg_sizes.raw(), num_block_rows);
            cudaCheckError();

            // Reassign nodes from oversized aggregates
            cudaMemsetAsync(d_reassigned.raw(), 0, sizeof(int), str);
            refine_aggregates_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                A.row_offsets.raw(), A.col_indices.raw(),
                edge_weights_orig.raw(), max_ew_per_row.raw(),
                aggregates.raw(), agg_sizes.raw(),
                this->m_max_aggregate_size, alpha,
                num_block_rows, d_reassigned.raw());
            cudaCheckError();

            int h_reassigned = 0;
            cudaMemcpyAsync(&h_reassigned, d_reassigned.raw(), sizeof(int),
                            cudaMemcpyDeviceToHost, str);
            cudaStreamSynchronize(str);

            if (h_reassigned == 0) break;

            amgx_printf("[MIS-k] Refine iter %d: reassigned %d nodes\n",
                        refine_iter, h_reassigned);
        }

        // Re-renumber after refinement (some aggregates may be empty now)
        this->renumberAndCountAggregates(aggregates, aggregates_global,
                                         num_block_rows, num_aggregates);

        amgx_printf("[MIS-k] After refinement (max_size=%d, threshold=%.2f): "
                    "%d aggregates (avg size %.1f)\n",
                    this->m_max_aggregate_size, alpha, num_aggregates,
                    (float)num_block_rows / (float)num_aggregates);
    }

    if (effective_k > 1)
    {
        amgx_printf("[MIS-k] Final: %d fine nodes -> %d aggregates "
                    "(net coarsening ratio %.2fx)\n",
                    num_block_rows, num_aggregates,
                    (float)num_block_rows / (float)num_aggregates);
    }
}

// -----------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------
#define AMGX_CASE_LINE(CASE) template class MISSelectorBase<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

#define AMGX_CASE_LINE(CASE) template class MISSelector<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace mis_selector
} // namespace aggregation
} // namespace amgx
