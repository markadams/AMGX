// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <aggregation/batched_qr.h>
#include <vector>
#include <cstdio>
#include <error.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust_wrapper.h>

namespace amgx
{
namespace aggregation
{

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// Maximum aggregate size (rows) supported: 27 nodes * 3 DOFs = 81
static const int MAX_AGG_ROWS = 128;
// Maximum null_dim supported
static const int MAX_NULL_DIM = 6;

// -----------------------------------------------------------------------
// Kernel: batched_qr_kernel
//
// One block per aggregate.  blockDim.x >= agg_size (padded to warp).
// Shared memory layout:
//   M[MAX_AGG_ROWS][MAX_NULL_DIM]  — local dense matrix, modified in-place
//   norms[MAX_NULL_DIM]            — column norms (scratch)
//   dots[MAX_NULL_DIM]             — dot products (scratch)
//
// Algorithm (Modified Gram-Schmidt):
//   for k = 0 .. null_dim-1:
//     norm = sqrt( sum_i M[i][k]^2 )
//     R[a, k, k] = norm
//     M[:,k] /= norm          (normalize column k -> Q[:,k])
//     for j = k+1 .. null_dim-1:
//       dot = sum_i M[i][k] * M[i][j]
//       R[a, k, j] = dot
//       M[:,j] -= dot * M[:,k]  (orthogonalize)
//
// After the loop, M[:,k] = Q[:,k] and R is upper-triangular.
// -----------------------------------------------------------------------

template <typename ValueType>
__global__
void batched_qr_kernel(int num_aggs,
                       int null_dim,
                       const int    * __restrict__ agg_row_offsets,
                       const int    * __restrict__ agg_rows,
                       const ValueType * __restrict__ B,
                       int num_rows,
                       ValueType    * __restrict__ P_tent_vals,
                       ValueType    * __restrict__ R_out)
{
    // Each block handles one aggregate
    int agg = blockIdx.x;
    if (agg >= num_aggs) return;

    int row_start = agg_row_offsets[agg];
    int row_end   = agg_row_offsets[agg + 1];
    int agg_size  = row_end - row_start;  // number of fine rows in this agg

    // Shared memory: M matrix + scratch arrays
    __shared__ ValueType M[MAX_AGG_ROWS][MAX_NULL_DIM];
    __shared__ ValueType sh_norm[MAX_NULL_DIM];
    __shared__ ValueType sh_dot[MAX_NULL_DIM];

    int tid = threadIdx.x;

    // ---- Load B rows for this aggregate into M ----
    // M[local_row][k] = B[k * num_rows + global_row]
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            M[local_row][k] = B[k * num_rows + global_row];
    }
    __syncthreads();

    // ---- Modified Gram-Schmidt QR ----
    for (int k = 0; k < null_dim; ++k)
    {
        // Step 1: compute norm of column k
        // Each thread accumulates partial sum for its rows
        ValueType partial = static_cast<ValueType>(0);
        for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
            partial += M[local_row][k] * M[local_row][k];

        // Reduce partial sums into sh_norm[k] via atomicAdd
        if (tid == 0) sh_norm[k] = static_cast<ValueType>(0);
        __syncthreads();
        atomicAdd(&sh_norm[k], partial);
        __syncthreads();

        ValueType norm = sqrt(sh_norm[k]);
        // Guard against zero column (degenerate aggregate)
        if (norm < static_cast<ValueType>(1e-14)) norm = static_cast<ValueType>(1);

        // Store R[a, k, k] = norm  (column-major: R[a][col][row] = R_out[a*nd*nd + col*nd + row])
        if (tid == 0)
            R_out[agg * null_dim * null_dim + k * null_dim + k] = norm;

        // Step 2: normalize column k
        ValueType inv_norm = static_cast<ValueType>(1) / norm;
        for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
            M[local_row][k] *= inv_norm;
        __syncthreads();

        // Step 3: orthogonalize remaining columns j > k
        for (int j = k + 1; j < null_dim; ++j)
        {
            // Compute dot product <Q[:,k], M[:,j]>
            ValueType pdot = static_cast<ValueType>(0);
            for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
                pdot += M[local_row][k] * M[local_row][j];

            if (tid == 0) sh_dot[j] = static_cast<ValueType>(0);
            __syncthreads();
            atomicAdd(&sh_dot[j], pdot);
            __syncthreads();

            // Store R[a, k, j]  (upper triangle, column-major: col=j, row=k)
            if (tid == 0)
                R_out[agg * null_dim * null_dim + j * null_dim + k] = sh_dot[j];

            // Subtract projection: M[:,j] -= dot * Q[:,k]
            ValueType dot = sh_dot[j];
            for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
                M[local_row][j] -= dot * M[local_row][k];
            __syncthreads();
        }
    }

    // ---- Write Q (= M after MGS) to P_tent_vals ----
    // P_tent_vals[global_row * null_dim + k] = M[local_row][k]
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            P_tent_vals[global_row * null_dim + k] = M[local_row][k];
    }
}

// -----------------------------------------------------------------------
// build_agg_row_lists
//
// Converts aggregates[i] -> (agg_row_offsets, agg_rows) CSR representation.
// Uses thrust for portability.
// -----------------------------------------------------------------------

void build_agg_row_lists(int num_rows,
                         int num_aggs,
                         const int *aggregates,
                         int *agg_row_offsets,
                         int *agg_rows)
{
    // Count rows per aggregate
    thrust::device_ptr<const int> agg_ptr(aggregates);
    thrust::device_ptr<int> offsets_ptr(agg_row_offsets);
    thrust::device_ptr<int> rows_ptr(agg_rows);

    // Zero offsets
    thrust::fill(offsets_ptr, offsets_ptr + num_aggs + 1, 0);

    // Count: offsets[agg+1]++
    // Use a temporary count array then exclusive_scan
    thrust::device_vector<int> counts(num_aggs, 0);
    // Histogram: count[aggregates[i]]++ for each i
    thrust::for_each(agg_ptr, agg_ptr + num_rows,
        [counts_raw = thrust::raw_pointer_cast(counts.data())] __device__ (int agg_id) {
            atomicAdd(counts_raw + agg_id, 1);
        });

    // Exclusive scan to get offsets: write to offsets[0..num_aggs-1],
    // then set offsets[num_aggs] = num_rows.
    // NOTE: must write to offsets_ptr (not offsets_ptr+1) to avoid off-by-one.
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets_ptr);
    thrust::device_ptr<int> last_offset(agg_row_offsets + num_aggs);
    *last_offset = num_rows;

    // Fill agg_rows: for each fine row i, place i at position offsets[agg]+local_idx
    // Use a second pass with atomic counters
    thrust::device_vector<int> cursor(num_aggs);
    thrust::copy(offsets_ptr + 1, offsets_ptr + num_aggs + 1, cursor.begin());
    // cursor[a] = offsets[a+1] initially; we'll use offsets[a] as base and atomicAdd
    // Reset cursor to offsets[a]
    thrust::copy(offsets_ptr, offsets_ptr + num_aggs, cursor.begin());

    int *cursor_raw = thrust::raw_pointer_cast(cursor.data());
    int *rows_raw   = thrust::raw_pointer_cast(rows_ptr);
    const int *agg_raw = aggregates;

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_rows),
        [agg_raw, cursor_raw, rows_raw] __device__ (int i) {
            int a   = agg_raw[i];
            int pos = atomicAdd(cursor_raw + a, 1);
            rows_raw[pos] = i;
        });
}

// -----------------------------------------------------------------------
// batched_qr — host launcher
// -----------------------------------------------------------------------

template <typename ValueType>
void batched_qr(int num_aggs,
                int null_dim,
                int num_rows,
                const int    *agg_row_offsets,
                const int    *agg_rows,
                const ValueType *B,
                ValueType    *P_tent_vals,
                ValueType    *R_out)
{
    if (num_aggs == 0 || null_dim == 0) return;

    if (null_dim > MAX_NULL_DIM)
        FatalError("batched_qr: null_dim exceeds MAX_NULL_DIM (6)", AMGX_ERR_BAD_PARAMETERS);


    // One block per aggregate; threads = min(MAX_AGG_ROWS, 128)
    int threads = 128;
    int blocks  = num_aggs;

    batched_qr_kernel<ValueType><<<blocks, threads>>>(
        num_aggs, null_dim,
        agg_row_offsets, agg_rows,
        B, num_rows,
        P_tent_vals, R_out);
    cudaCheckError();
}

// -----------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------

template void batched_qr<float>(int, int, int, const int *, const int *,
                                const float *, float *, float *);
template void batched_qr<double>(int, int, int, const int *, const int *,
                                 const double *, double *, double *);

} // namespace aggregation
} // namespace amgx
