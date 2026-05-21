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
//       Run MIS-1 on A_cur (with exchange_halo for MPI correctness)
//       Compose R_out_agg via thrust::gather
//       If not last pass: A_cur = R * A_cur * R^T  (Galerkin product)
//         For multi-GPU: distributed RAP via initialize_manager,
//         exchange_halo_rows_P, exchange_RAP_ext, renumberMatrixOneRing,
//         and createOneRingHaloRows so A_cur is a proper distributed
//         matrix for the next pass.
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
#include <distributed/distributed_manager.h>
#include <distributed/distributed_arranger.h>

#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <cusp/detail/format_utils.h>  // offsets_to_indices

#include <vector>
#include <cmath>
#include <climits>

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
__global__ __launch_bounds__(256, 8)
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
__global__ __launch_bounds__(256, 4)
void find_mis_k1_kernel(const IndexType *row_offsets,
                        const IndexType *col_indices,
                        const float     *node_weights,
                        const float     *edge_weights,
                        const float      strength_threshold,
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

                // Skip weak edges if strength threshold is active
                if (strength_threshold > 0.0f && edge_weights[j] < strength_threshold)
                { continue; }

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
__global__ __launch_bounds__(256, 2)
void find_mis_k2_kernel(const IndexType *row_offsets,
                        const IndexType *col_indices,
                        const float     *node_weights,
                        const float     *edge_weights,
                        const float      strength_threshold,
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
            bool dominated  = false;
            bool is_max     = true;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            // Check distance-1 neighbors
            for (int j = jmin; j < jmax && !dominated; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= total_rows) { continue; }

                // Skip weak edges if strength threshold is active
                if (strength_threshold > 0.0f && edge_weights[j] < strength_threshold)
                { continue; }

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
                // Expand any node with valid row_offsets (owned + halo).
                // When the matrix has 2-ring halo rows, row_offsets are valid
                // for 1-ring halo nodes (jcol < total_rows), allowing the
                // kernel to reach distance-2 nodes across partition boundaries.
                if (jcol < total_rows)
                {
                    int kmin = row_offsets[jcol];
                    int kmax = row_offsets[jcol + 1];
                    for (int k = kmin; k < kmax && !dominated; k++)
                    {
                        int kcol = col_indices[k];
                        if (kcol == tid || kcol == jcol || kcol >= total_rows) { continue; }

                        // Skip weak edges at distance-2
                        if (strength_threshold > 0.0f && edge_weights[k] < strength_threshold)
                        { continue; }

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
__global__ __launch_bounds__(256, 4)
void assign_aggregates_kernel(const IndexType *row_offsets,
                              const IndexType *col_indices,
                              const float     *edge_weights,
                              const float      strength_threshold,
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

                // Skip weak edges during assignment
                if (strength_threshold > 0.0f && edge_weights[j] < strength_threshold)
                { continue; }

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
//   Find the best neighbor aggregate that is below the size cap
//   using score = edge_weight / (target_agg_size + 1).
//
// The root node (tid == my_agg) is never moved to preserve aggregate identity.
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
        int my_size = (my_agg >= 0) ? agg_sizes[my_agg] : 0;
        // Only process nodes in oversized aggregates; skip the root node
        if (my_agg >= 0 && my_size > max_agg_size && tid != my_agg)
        {
            float threshold = max_edge_weight[tid] * alpha;
            float best_score = -1.0f;
            int   best_agg   = -1;

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
                    int jagg_size = agg_sizes[jagg];
                    // Only move to aggregates below the size cap
                    if (jagg_size < max_agg_size)
                    {
                        float score = w / (float)(jagg_size + 1);
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
                aggregates[tid] = best_agg;
                atomicAdd(num_reassigned, 1);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernel: split_oversized_kernel
//
// For aggregates still oversized after refinement, split off boundary nodes
// to create new singletons. Uses a deterministic selection: for each
// oversized aggregate, the boundary node whose (tid % agg_size) == iter
// becomes a singleton. This splits off ~1 node per oversized aggregate
// per call, controlled by the iter parameter.
// -----------------------------------------------------------------------
template <typename IndexType>
__global__
void split_oversized_kernel(
    const IndexType *row_offsets,
    const IndexType *col_indices,
    IndexType       *aggregates,
    const int       *agg_sizes,
    const int        max_agg_size,
    const int        num_rows,
    const int        iter,
    int             *num_split)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)
    {
        int my_agg = aggregates[tid];
        int my_size = (my_agg >= 0) ? agg_sizes[my_agg] : 0;
        // Only process non-root nodes in oversized aggregates
        if (my_agg >= 0 && tid != my_agg && my_size > max_agg_size)
        {
            // Deterministic selection: pick ~1 node per aggregate per iter
            if ((tid % my_size) == (iter % my_size))
            {
                // Verify this node is at an aggregate boundary
                bool at_boundary = false;
                int jmin = row_offsets[tid];
                int jmax = row_offsets[tid + 1];
                for (int j = jmin; j < jmax; j++)
                {
                    int jcol = col_indices[j];
                    if (jcol < num_rows && aggregates[jcol] != my_agg)
                    {
                        at_boundary = true;
                        break;
                    }
                }

                if (at_boundary)
                {
                    aggregates[tid] = tid;
                    atomicAdd(num_split, 1);
                }
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}

// -----------------------------------------------------------------------
// Kernels for distributed column compression (MIS-k Galerkin loop)
// Mirrors sa_flag_halo_columns_kernel / sa_compress_halo_columns_kernel /
// sa_compress_l2g_kernel from aggregation_amg_level.cu.
// -----------------------------------------------------------------------

// Flag which halo columns (col >= nrow) are referenced in the owned rows.
template <typename IndexType>
__global__
void misk_flag_halo_columns_kernel(IndexType nrow, const IndexType *row_offsets,
                                    const IndexType *col_indices, IndexType *flags)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrow;
         row += gridDim.x * blockDim.x)
    {
        int s = row_offsets[row];
        int e = row_offsets[row + 1];
        for (int j = s; j < e; j++)
        {
            IndexType col = col_indices[j];
            if (col >= nrow)
            {
                flags[col - nrow] = 1;
            }
        }
    }
}

// Remap halo column indices using the prefix-summed flags array.
template <typename IndexType>
__global__
void misk_compress_halo_columns_kernel(IndexType nrow, const IndexType *row_offsets,
                                        IndexType *col_indices, const IndexType *flags)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < nrow;
         row += gridDim.x * blockDim.x)
    {
        int s = row_offsets[row];
        int e = row_offsets[row + 1];
        for (int j = s; j < e; j++)
        {
            IndexType col = col_indices[j];
            if (col >= nrow)
            {
                col_indices[j] = nrow + flags[col - nrow];
            }
        }
    }
}

// Compress local_to_global_map: keep only entries where flags[i] != flags[i+1].
template <typename IndexType, typename Int64Type>
__global__
void misk_compress_l2g_kernel(IndexType nl2g, const Int64Type *l2g_in,
                               Int64Type *l2g_out, const IndexType *flags)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nl2g;
         i += gridDim.x * blockDim.x)
    {
        if (flags[i] != flags[i + 1])
        {
            l2g_out[flags[i]] = l2g_in[i];
        }
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
    m_max_aggregate_size = cfg.AMG_Config::template getParameter<int>("max_aggregate_size", cfg_scope);
    m_refine_threshold = cfg.AMG_Config::template getParameter<double>("refine_threshold", cfg_scope);
    m_mis2_algorithm = cfg.AMG_Config::template getParameter<int>("mis2_algorithm", cfg_scope);
    m_strength_threshold = cfg.AMG_Config::template getParameter<double>("strength_threshold", cfg_scope);
    m_verbose = cfg.AMG_Config::template getParameter<int>("mis_verbose", cfg_scope);
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
    // Determine effective MIS distance (k).
    // Multi-GPU MIS-k is supported via the distributed Galerkin loop
    // in Phase 6 below (initialize_manager, exchange_halo_rows_P,
    // exchange_RAP_ext, renumberMatrixOneRing, createOneRingHaloRows).
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

    // Multi-GPU MIS-k is now supported via the distributed Galerkin loop below.
    // No fallback needed.

    if (this->m_verbose)
    {
        amgx_printf("[MIS-k] Level %d: mis_k=%d, effective_k=%d "
                    "(aggressive_levels=%d)\n",
                    current_level, this->m_mis_k, effective_k,
                    this->m_aggressive_levels);
    }

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

    // ==================================================================
    // Algorithm selection: implicit square graph MIS-2 vs Galerkin loop
    // ==================================================================
    if (this->m_mis2_algorithm == 1 && effective_k == 2)
    {
        // --------------------------------------------------------------
        // Implicit square graph path (mis2_algorithm=1):
        //   1. Run MIS-2 directly on the original graph
        //   2. Assign aggregates using distance-1 connectivity only
        //   3. Propagate to unassigned nodes
        //   4. Renumber
        //
        // This produces smaller aggregates than Galerkin MIS-2 because
        // distance-1 assignment only grabs immediate neighbors (~4 for
        // 5-pt stencil) rather than the dense Galerkin neighborhood.
        // --------------------------------------------------------------
        if (this->m_verbose)
            amgx_printf("[MIS-k] Using implicit square graph MIS-2 (mis2_algorithm=1)\n");

        // ------------------------------------------------------------------
        // Multi-GPU: MIS-2 (Algorithm 1) requires 2-ring halo data so that
        // the find_mis_k2_kernel can expand distance-2 neighbors through
        // halo nodes near partition boundaries.  Enforce that the matrix was
        // set up with at least 2 halo rings.
        // ------------------------------------------------------------------
        if (!A.is_matrix_singleGPU())
        {
            int nrings = A.manager->num_halo_rings();
            if (nrings < 2)
            {
                FatalError("MIS-2 Algorithm 1 (implicit square graph) in multi-GPU mode "
                           "requires num_rings >= 2, but the matrix has only 1 ring. "
                           "Set communicator_num_rings=2 in your AMGx config.",
                           AMGX_ERR_CONFIGURATION);
            }
            else if (this->m_verbose)
            {
                amgx_printf("[MIS-k] Matrix has %d halo ring(s) — "
                            "2-ring halo exchange enabled for MIS-2\n", nrings);
            }
        }

        const int num_blocks_fine = std::min(AMGX_GRID_MAX_SIZE,
                                             (num_block_rows - 1) / threads_per_block + 1);

        // Strength threshold (absolute comparison against normalized edge weight)
        float strength_thresh = (float)this->m_strength_threshold;
        if (this->m_verbose && strength_thresh > 0.0f)
            amgx_printf("[MIS-k] Strength threshold: %.4f (absolute)\n", strength_thresh);

        // Phase 1: Assign node weights
        unsigned int seed = 0x9e3779b9u;
        FVector_d node_weights(total_rows, 0.0f);
        assign_node_weights_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
            num_block_rows, node_weights.raw(), seed);
        cudaCheckError();

        // Exchange node_weights halo for MPI (2-ring)
        // Two successive 1-ring exchanges propagate data to distance-2 halo
        // nodes: first exchange fills ring-1 halo entries, second exchange
        // uses those ring-1 values to fill ring-2 halo entries.
        if (!A.is_matrix_singleGPU())
        {
            node_weights.dirtybit = 1;
            A.manager->exchange_halo(node_weights, node_weights.tag);
            // Second exchange: propagate ring-1 halo values to ring-2
            node_weights.dirtybit = 1;
            A.manager->exchange_halo(node_weights, node_weights.tag);
        }

        // Phase 2: Run MIS-2 iteratively until converged
        IVector_d status_vec(total_rows, MIS_UNDECIDED);
        int *status_ptr = status_vec.raw();

        IVector_d d_changed(1);
        int *d_changed_ptr = d_changed.raw();

        int num_undecided = num_block_rows;
        int icount = 0;

        cudaFuncSetCacheConfig(find_mis_k2_kernel<IndexType>, cudaFuncCachePreferL1);

        while (num_undecided > 0 && icount < this->m_max_iterations)
        {
            cudaMemsetAsync(d_changed_ptr, 0, sizeof(int), str);

            find_mis_k2_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                A.row_offsets.raw(), A.col_indices.raw(),
                node_weights.raw(), edge_weights_orig.raw(),
                strength_thresh,
                status_ptr,
                num_block_rows, total_rows, d_changed_ptr);
            cudaCheckError();

            // Exchange status halo for MPI (2-ring)
            // Two successive 1-ring exchanges: first fills ring-1 halo
            // entries with MIS status from neighboring partitions, second
            // propagates those to ring-2 halo entries so the kernel can
            // check distance-2 independence across partition boundaries.
            if (!A.is_matrix_singleGPU())
            {
                status_vec.dirtybit = 1;
                A.manager->exchange_halo(status_vec, status_vec.tag);
                // Second exchange: ring-1 → ring-2
                status_vec.dirtybit = 1;
                A.manager->exchange_halo(status_vec, status_vec.tag);
            }

            num_undecided = (int)thrust_wrapper::count<AMGX_device>(
                status_vec.begin(), status_vec.begin() + num_block_rows,
                MIS_UNDECIDED);
            cudaCheckError();
            icount++;
        }

        // Safety: remaining undecided become roots
        if (num_undecided > 0)
        {
            thrust::transform_if(
                thrust::cuda::par.on(str),
                status_vec.begin(), status_vec.begin() + num_block_rows,
                status_vec.begin(),
                SetMISRoot(),
                IsUndecided());
            cudaCheckError();
        }

        int n_roots_mis = (int)thrust_wrapper::count<AMGX_device>(
            status_vec.begin(), status_vec.begin() + num_block_rows, MIS_ROOT);
        cudaCheckError();
        if (this->m_verbose)
            amgx_printf("[MIS-k] Implicit MIS-2: converged in %d iters, "
                        "%d roots out of %d nodes (%.1f%%), %d forced-root\n",
                        icount, n_roots_mis, num_block_rows,
                        100.0f * n_roots_mis / num_block_rows,
                        num_undecided);

        // Check coarsening ratio BEFORE aggregate assignment to avoid crashes
        float coarsening_ratio = (float)num_block_rows / (float)n_roots_mis;
        if (coarsening_ratio > 64.0f)
        {
            // Too aggressive — fall back to MIS-1 on the original graph
            amgx_printf("[MIS-k] Implicit MIS-2: %d roots from %d nodes (ratio %.1fx) "
                        "— too aggressive, falling back to MIS-1\n",
                        n_roots_mis, num_block_rows, coarsening_ratio);

            // Re-run with MIS-1
            unsigned int seed_fb = 0x9e3779b9u + 0x12345678u;
            FVector_d node_weights_fb(total_rows, 0.0f);
            assign_node_weights_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                num_block_rows, node_weights_fb.raw(), seed_fb);
            cudaCheckError();

            IVector_d status_fb(total_rows, MIS_UNDECIDED);
            IVector_d d_changed_fb(1);
            int num_undecided_fb = num_block_rows;
            int icount_fb = 0;
            while (num_undecided_fb > 0 && icount_fb < this->m_max_iterations)
            {
                cudaMemsetAsync(d_changed_fb.raw(), 0, sizeof(int), str);
                find_mis_k1_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                    A.row_offsets.raw(), A.col_indices.raw(),
                    node_weights_fb.raw(), edge_weights_orig.raw(),
                    strength_thresh,
                    status_fb.raw(),
                    num_block_rows, total_rows, d_changed_fb.raw());
                cudaCheckError();
                num_undecided_fb = (int)thrust_wrapper::count<AMGX_device>(
                    status_fb.begin(), status_fb.begin() + num_block_rows,
                    MIS_UNDECIDED);
                cudaCheckError();
                icount_fb++;
            }
            if (num_undecided_fb > 0)
            {
                thrust::transform_if(thrust::cuda::par.on(str),
                    status_fb.begin(), status_fb.begin() + num_block_rows,
                    status_fb.begin(), SetMISRoot(), IsUndecided());
                cudaCheckError();
            }

            // Use MIS-1 status for aggregate assignment
            status_ptr = status_fb.raw();
        }

        // Phase 3: Assign non-MIS nodes to strongest root neighbor (distance-1)
        thrust_wrapper::fill<AMGX_device>(
            aggregates.begin(), aggregates.begin() + num_block_rows, -1);
        assign_aggregates_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
            A.row_offsets.raw(), A.col_indices.raw(),
            edge_weights_orig.raw(), strength_thresh,
            status_ptr,
            aggregates.raw(), num_block_rows);
        cudaCheckError();

        // Phase 4: Propagate to unassigned nodes
        int num_unassigned = (int)thrust_wrapper::count<AMGX_device>(
            aggregates.begin(), aggregates.begin() + num_block_rows, -1);
        cudaCheckError();

        if (num_unassigned > 0)
        {
            if (!this->m_merge_singletons)
            {
                make_singletons_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                    aggregates.raw(), num_block_rows);
                cudaCheckError();
            }
            else
            {
                IVector_d agg_candidate(num_block_rows, -1);
                while (num_unassigned > 0)
                {
                    thrust_wrapper::fill<AMGX_device>(
                        agg_candidate.begin(), agg_candidate.end(), -1);
                    propagate_aggregates_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                        A.row_offsets.raw(), A.col_indices.raw(),
                        edge_weights_orig.raw(), aggregates.raw(),
                        agg_candidate.raw(), num_block_rows, /*deterministic=*/1);
                    cudaCheckError();
                    join_candidates_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                        aggregates.raw(), agg_candidate.raw(), num_block_rows);
                    cudaCheckError();
                    int prev_unassigned = num_unassigned;
                    num_unassigned = (int)thrust_wrapper::count<AMGX_device>(
                        aggregates.begin(), aggregates.begin() + num_block_rows, -1);
                    cudaCheckError();
                    if (num_unassigned == prev_unassigned)
                    {
                        make_singletons_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                            aggregates.raw(), num_block_rows);
                        cudaCheckError();
                        break;
                    }
                }
            }
        }

        // Phase 5: Renumber aggregates
        this->renumberAndCountAggregates(aggregates, aggregates_global,
                                         num_block_rows, num_aggregates);

        if (this->m_verbose)
            amgx_printf("[MIS-k] Implicit MIS-2: %d fine nodes -> %d aggregates "
                        "(avg size %.1f, net coarsening ratio %.2fx)\n",
                        num_block_rows, num_aggregates,
                        (float)num_block_rows / (float)num_aggregates,
                    (float)num_block_rows / (float)num_aggregates);
    }
    else
    {
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

        if (effective_k > 1 && this->m_verbose)
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

        // Strength threshold (absolute comparison against normalized edge weight)
        float strength_thresh = (float)this->m_strength_threshold;

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

        cudaFuncSetCacheConfig(find_mis_k1_kernel<IndexType>, cudaFuncCachePreferL1);

        while (num_undecided > 0 && icount < this->m_max_iterations)
        {
            cudaMemsetAsync(d_changed_ptr, 0, sizeof(int), str);

            // Always use find_mis_k1_kernel in the Galerkin loop;
            // the k-distance is achieved by iterating the loop effective_k times.
            find_mis_k1_kernel<<<num_blocks_cur, threads_per_block, 0, str>>>(
                A_cur.row_offsets.raw(), A_cur.col_indices.raw(),
                node_weights.raw(), edge_weights_cur_ptr->raw(),
                strength_thresh,
                status_ptr,
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
        if (effective_k > 1 && this->m_verbose)
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
            edge_weights_cur_ptr->raw(), strength_thresh,
            status_ptr,
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

        if (effective_k > 1 && this->m_verbose)
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
        //
        // For multi-GPU: follow the distributed RAP pattern from
        // createCoarseMatrices() in aggregation_amg_level.cu:
        //   1. initialize_manager() on P to set up coarse part_offsets
        //   2. exchange_halo_rows_P() to get halo rows of P from neighbors
        //   3. Local RAP = R * A_scalar * P (including halo contributions)
        //   4. exchange_RAP_ext() to assemble off-process contributions
        //   5. renumberMatrixOneRing() + createOneRingHaloRows() to finalize
        // ----------------------------------------------------------------
        if (pass < effective_k - 1)
        {
            const bool is_distributed = !A_cur.is_matrix_singleGPU();
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
            // For distributed: A_scalar shares A_cur's sparsity and manager
            if (is_distributed)
            {
                A_scalar.manager = A_cur.manager;
                A_scalar.setView(OWNED);
            }
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

            if (!is_distributed)
            {
                // ============================================================
                // Single-GPU path (unchanged)
                // ============================================================
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
                    nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
                A_coarse.computeDiagonal();
                A_coarse.set_initialized(1);
            }
            else
            {
                // ============================================================
                // Multi-GPU distributed RAP path
                //
                // Follows the SA distributed RAP pattern from
                // createCoarseMatrices() in aggregation_amg_level.cu.
                // ============================================================
                typedef TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec> TConfig_h;
                typedef typename TConfig_d::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_d;
                typedef typename TConfig_h::template setVecPrec<AMGX_vecInt64>::Type i64vec_value_type_h;
                typedef typename TConfig_h::template setVecPrec<AMGX_vecInt>::Type ivec_value_type_h;
                typedef Vector<i64vec_value_type_d> I64Vector;
                typedef Vector<i64vec_value_type_h> I64Vector_h;
                typedef Vector<ivec_value_type_h> IVector_h;

                int num_owned_coarse_pts = n_roots;

                // --- Step 1: Set up P_cur's manager via initialize_manager ---
                // This does an allgather to compute part_offsets for the
                // coarse level (how many coarse DOFs each rank owns).
                DistributedArranger<TConfig_d> *prep =
                    new DistributedArranger<TConfig_d>;
                prep->initialize_manager(A_scalar, P_cur, num_owned_coarse_pts);

                // --- Step 2: Set up A_coarse.manager ---
                IndexType my_rank = A_cur.manager->global_id();

                A_coarse.set_initialized(0);
                if (A_coarse.manager == NULL)
                {
                    A_coarse.manager = new DistributedManager<TConfig_d>();
                }
                A_coarse.manager->A = &A_coarse;
                A_coarse.manager->setComms(A_cur.manager->getComms());
                A_coarse.manager->set_global_id(my_rank);
                A_coarse.manager->set_num_partitions(
                    A_cur.manager->get_num_partitions());
                A_coarse.manager->part_offsets_h =
                    P_cur.manager->part_offsets_h;
                A_coarse.manager->part_offsets =
                    P_cur.manager->part_offsets;
                A_coarse.manager->set_base_index(
                    A_coarse.manager->part_offsets_h[my_rank]);
                A_coarse.manager->set_index_range(num_owned_coarse_pts);
                A_coarse.manager->num_rows_global =
                    A_coarse.manager->part_offsets_h[
                        A_cur.manager->get_num_partitions()];
                A_coarse.manager->local_to_global_map.resize(0);

                // --- Step 3: Exchange halo rows of P ---
                // Save P manager info for exchange_RAP_ext later
                IVector_h P_neighbors_h = P_cur.manager->neighbors;
                I64Vector_h P_halo_ranges_h = P_cur.manager->halo_ranges_h;
                I64Vector P_halo_ranges = P_cur.manager->halo_ranges;
                IVector_h P_halo_offsets_h = P_cur.manager->halo_offsets;

                prep->exchange_halo_rows_P(
                    A_scalar, P_cur,
                    A_coarse.manager->local_to_global_map,
                    P_neighbors_h, P_halo_ranges_h, P_halo_ranges,
                    P_halo_offsets_h,
                    A_coarse.manager->part_offsets_h,
                    A_coarse.manager->part_offsets,
                    num_owned_coarse_pts,
                    A_coarse.manager->part_offsets_h[my_rank]);
                cudaCheckError();
                delete prep;

                // --- Step 4: Compute R = P^T (owned rows only) ---
                // Set P_cur to show only owned fine rows for the transpose
                int num_owned_fine_pts = A_cur.manager->halo_offsets[0];
                P_cur.set_initialized(0);
                P_cur.set_num_rows(num_owned_fine_pts);
                P_cur.addProps(CSR);
                P_cur.set_initialized(1);

                Matrix_d R_cur;
                R_cur.set_initialized(0);
                R_cur.addProps(CSR);
                R_cur.delProps(COO);
                R_cur.delProps(DIAG);
                R_cur.setColsReorderedByColor(false);
                transpose(P_cur, R_cur);

                // --- Step 5: Compute RAP_full = R * A_scalar * P ---
                // (local computation including halo contributions)
                Matrix_d RAP_full;
                RAP_full.set_initialized(0);
                A_scalar.setView(OWNED);
                CSR_Multiply<TConfig_d>::csr_galerkin_product(
                    R_cur, A_scalar, P_cur, RAP_full,
                    nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr);
                RAP_full.set_initialized(1);

                // --- Step 6: Exchange RAP ext ---
                // Update P_cur.manager with info modified by
                // exchange_halo_rows_P
                P_cur.manager->neighbors = P_neighbors_h;
                P_cur.manager->halo_offsets = P_halo_offsets_h;
                P_cur.manager->halo_ranges_h = P_halo_ranges_h;
                P_cur.manager->halo_ranges = P_halo_ranges;

                DistributedArranger<TConfig_d> *prep2 =
                    new DistributedArranger<TConfig_d>;
                prep2->exchange_RAP_ext(
                    A_coarse, RAP_full, A_scalar, P_cur,
                    P_halo_offsets_h,
                    A_coarse.manager->local_to_global_map,
                    P_neighbors_h, P_halo_ranges_h, P_halo_ranges,
                    A_coarse.manager->part_offsets_h,
                    A_coarse.manager->part_offsets,
                    num_owned_coarse_pts,
                    A_coarse.manager->part_offsets_h[my_rank],
                    nullptr);
                delete prep2;

                // --- Step 7: Column compression ---
                // Remove unused halo columns from A_coarse
                IndexType nrow_c = A_coarse.get_num_rows();
                IndexType ncol_c = A_coarse.get_num_cols();
                IndexType nl2g = ncol_c - nrow_c;
                if (nl2g > 0)
                {
                    IVector_d l2g_p(nl2g + 1, 0);
                    I64Vector l2g_t(nl2g, 0);
                    IndexType nblocks_cc = std::min(4096,
                        (int)((nrow_c + 127) / 128));

                    // Flag referenced halo columns
                    if (nblocks_cc > 0)
                    {
                        misk_flag_halo_columns_kernel<IndexType>
                            <<<nblocks_cc, 128, 0, str>>>(
                            nrow_c, A_coarse.row_offsets.raw(),
                            A_coarse.col_indices.raw(), l2g_p.raw());
                        cudaCheckError();
                    }

                    // Exclusive scan to get new positions
                    thrust_wrapper::exclusive_scan<AMGX_device>(
                        l2g_p.begin(), l2g_p.end(), l2g_p.begin());
                    cudaCheckError();

                    // Read new count
                    int new_nl2g_val = 0;
                    cudaMemcpy(&new_nl2g_val,
                               l2g_p.raw() + nl2g, sizeof(int),
                               cudaMemcpyDeviceToHost);

                    if (new_nl2g_val < nl2g)
                    {
                        // Compress column indices
                        if (nblocks_cc > 0)
                        {
                            misk_compress_halo_columns_kernel<IndexType>
                                <<<nblocks_cc, 128, 0, str>>>(
                                nrow_c, A_coarse.row_offsets.raw(),
                                A_coarse.col_indices.raw(),
                                l2g_p.raw());
                            cudaCheckError();
                        }

                        // Compress local_to_global_map
                        IndexType nblocks_l2g = std::min(4096,
                            (int)((nl2g + 127) / 128));
                        if (nblocks_l2g > 0)
                        {
                            misk_compress_l2g_kernel<IndexType, int64_t>
                                <<<nblocks_l2g, 128, 0, str>>>(
                                nl2g,
                                A_coarse.manager->
                                    local_to_global_map.raw(),
                                l2g_t.raw(), l2g_p.raw());
                            cudaCheckError();
                        }

                        // Update matrix dimensions
                        A_coarse.set_initialized(0);
                        A_coarse.set_num_cols(nrow_c + new_nl2g_val);
                        A_coarse.set_initialized(1);

                        // Copy compressed l2g
                        amgx::thrust::copy(
                            l2g_t.begin(),
                            l2g_t.begin() + new_nl2g_val,
                            A_coarse.manager->
                                local_to_global_map.begin());
                        cudaCheckError();
                        A_coarse.manager->
                            local_to_global_map.resize(new_nl2g_val);
                    }
                }

                // --- Step 8: Finalize distributed coarse matrix ---
                A_coarse.set_initialized(0);
                A_coarse.manager->renumberMatrixOneRing(0);
                A_coarse.manager->createOneRingHaloRows();
                A_coarse.manager->getComms()->set_neighbors(
                    A_coarse.manager->num_neighbors());
                A_coarse.setView(OWNED);
                A_coarse.computeDiagonal();
                A_coarse.set_initialized(1);

                // Detach A_scalar's borrowed manager so it won't be
                // double-freed when A_scalar goes out of scope
                A_scalar.manager = NULL;
            }

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

            if (effective_k > 1 && this->m_verbose)
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

        IVector_d agg_sizes(num_block_rows);
        IVector_d d_reassigned(1);

        for (int refine_iter = 0; refine_iter < max_refine_iters; refine_iter++)
        {
            // Count aggregate sizes
            thrust_wrapper::fill<AMGX_device>(
                agg_sizes.begin(), agg_sizes.end(), 0);
            count_aggregate_sizes_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                aggregates.raw(), agg_sizes.raw(), num_block_rows);
            cudaCheckError();

            // Reassign nodes from oversized aggregates to smaller neighbors
            cudaMemsetAsync(d_reassigned.raw(), 0, sizeof(int), str);
            refine_aggregates_kernel<<<num_blocks_fine, threads_per_block, 0, str>>>(
                A.row_offsets.raw(), A.col_indices.raw(),
                edge_weights_orig.raw(), max_ew_per_row.raw(),
                aggregates.raw(), agg_sizes.raw(),
                this->m_max_aggregate_size, alpha,
                num_block_rows, d_reassigned.raw());
            cudaCheckError();

            // NOTE: split_oversized_kernel removed — it created singleton
            // aggregates (aggregates[tid]=tid) that are pathologically small
            // and disjointed, leading to dense coarse matrices and crashes.
            // The refine_aggregates_kernel above handles reassignment to
            // existing smaller neighbors without creating new aggregates.

            int h_reassigned = 0;
            cudaMemcpyAsync(&h_reassigned, d_reassigned.raw(), sizeof(int),
                            cudaMemcpyDeviceToHost, str);
            cudaStreamSynchronize(str);

            if (h_reassigned == 0) break;

            if (this->m_verbose)
                amgx_printf("[MIS-k] Refine iter %d: reassigned %d nodes\n",
                            refine_iter, h_reassigned);
        }

        // Re-renumber after refinement (some aggregates may be empty now)
        this->renumberAndCountAggregates(aggregates, aggregates_global,
                                         num_block_rows, num_aggregates);

        if (this->m_verbose)
            amgx_printf("[MIS-k] After refinement (max_size=%d, threshold=%.2f): "
                        "%d aggregates (avg size %.1f)\n",
                        this->m_max_aggregate_size, alpha, num_aggregates,
                        (float)num_block_rows / (float)num_aggregates);
    }

    // Aggregate size statistics (gated behind verbose flag to avoid D→H copy)
    if (this->m_verbose)
    {
        IVector_d agg_sizes_stats(num_aggregates, 0);
        const int num_blocks_stats = std::min(AMGX_GRID_MAX_SIZE,
                                              (num_block_rows - 1) / threads_per_block + 1);
        count_aggregate_sizes_kernel<<<num_blocks_stats, threads_per_block, 0, str>>>(
            aggregates.raw(), agg_sizes_stats.raw(), num_block_rows);
        cudaCheckError();

        // Copy to host for statistics
        std::vector<int> h_sizes(num_aggregates);
        cudaMemcpy(h_sizes.data(), agg_sizes_stats.raw(),
                   num_aggregates * sizeof(int), cudaMemcpyDeviceToHost);

        int min_size = INT_MAX, max_size = 0;
        double sum = 0.0, sum_sq = 0.0;
        int hist[6] = {0}; // [0]=size 1, [1]=2-3, [2]=4-5, [3]=6-7, [4]=8-9, [5]=10+
        for (int i = 0; i < num_aggregates; i++) {
            int s = h_sizes[i];
            if (s < min_size) min_size = s;
            if (s > max_size) max_size = s;
            sum += s;
            sum_sq += (double)s * s;
            if (s <= 1) hist[0]++;
            else if (s <= 3) hist[1]++;
            else if (s <= 5) hist[2]++;
            else if (s <= 7) hist[3]++;
            else if (s <= 9) hist[4]++;
            else hist[5]++;
        }
        double avg = sum / num_aggregates;
        double stddev = sqrt(sum_sq / num_aggregates - avg * avg);

        amgx_printf("[MIS-k] Aggregate stats: n=%d, min=%d, max=%d, avg=%.2f, stddev=%.2f\n",
                    num_aggregates, min_size, max_size, avg, stddev);
        amgx_printf("[MIS-k] Size histogram: [1]=%d [2-3]=%d [4-5]=%d [6-7]=%d [8-9]=%d [10+]=%d\n",
                    hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
    }

    if (effective_k > 1 && this->m_verbose)
    {
        amgx_printf("[MIS-k] Final: %d fine nodes -> %d aggregates "
                    "(net coarsening ratio %.2fx)\n",
                    num_block_rows, num_aggregates,
                    (float)num_block_rows / (float)num_aggregates);
    }
    } // end else (Galerkin path)
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
