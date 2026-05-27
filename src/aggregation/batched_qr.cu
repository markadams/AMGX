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
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust_wrapper.h>

namespace amgx
{
namespace aggregation
{

// -----------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------

// Maximum null_dim supported
static const int MAX_NULL_DIM = 6;

// -----------------------------------------------------------------------
// Kernel: batched_qr_kernel
//
// One block per aggregate.  Uses dynamic shared memory so there is no
// hard limit on aggregate size.
//
// Dynamic shared memory layout (all ValueType, contiguous):
//   M    [max_agg_rows * null_dim]  — local dense matrix (row-major), modified in-place
//   norm [null_dim]                 — column norms (scratch)
//   dot  [null_dim]                 — dot products (scratch)
//
// max_agg_rows is the maximum aggregate size across all aggregates,
// passed as a kernel parameter so each block can index into M correctly.
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
                       int max_agg_rows,
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

    // Dynamic shared memory layout:
    //   M[max_agg_rows * null_dim]  (row-major: M[local_row * null_dim + k])
    //   sh_norm[null_dim]
    //   sh_dot[null_dim]
    extern __shared__ char smem_raw[];
    ValueType *M       = reinterpret_cast<ValueType *>(smem_raw);
    ValueType *sh_norm = M + max_agg_rows * null_dim;
    ValueType *sh_dot  = sh_norm + null_dim;

    int tid = threadIdx.x;

    // ---- Load B rows for this aggregate into M ----
    // M[local_row * null_dim + k] = B[k * num_rows + global_row]
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            M[local_row * null_dim + k] = B[k * num_rows + global_row];
    }
    __syncthreads();

    // ---- Modified Gram-Schmidt QR ----
    for (int k = 0; k < null_dim; ++k)
    {
        // Step 1: compute norm of column k
        ValueType partial = static_cast<ValueType>(0);
        for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
        {
            ValueType v = M[local_row * null_dim + k];
            partial += v * v;
        }

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
            M[local_row * null_dim + k] *= inv_norm;
        __syncthreads();

        // Step 3: orthogonalize remaining columns j > k
        for (int j = k + 1; j < null_dim; ++j)
        {
            // Compute dot product <Q[:,k], M[:,j]>
            ValueType pdot = static_cast<ValueType>(0);
            for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
                pdot += M[local_row * null_dim + k] * M[local_row * null_dim + j];

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
                M[local_row * null_dim + j] -= dot * M[local_row * null_dim + k];
            __syncthreads();
        }
    }

    // ---- Write Q (= M after MGS) to P_tent_vals ----
    // P_tent_vals[global_row * null_dim + k] = M[local_row * null_dim + k]
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            P_tent_vals[global_row * null_dim + k] = M[local_row * null_dim + k];
    }
}

// -----------------------------------------------------------------------
// Global-memory MGS kernel (fallback: aggregate too large for smem)
//
// scratch: pre-allocated global buffer, size num_aggs * max_agg_rows * null_dim
//          Each block uses scratch + agg * max_agg_rows * null_dim as its M.
// smem: only sh_norm[null_dim] + sh_dot[null_dim] (tiny, always fits)
// -----------------------------------------------------------------------
template <typename ValueType>
__global__
void batched_qr_kernel_gmem(int num_aggs,
                             int null_dim,
                             int max_agg_rows,
                             const int    * __restrict__ agg_row_offsets,
                             const int    * __restrict__ agg_rows,
                             const ValueType * __restrict__ B,
                             int num_rows,
                             ValueType    * __restrict__ P_tent_vals,
                             ValueType    * __restrict__ R_out,
                             ValueType    * __restrict__ scratch)
{
    int agg = blockIdx.x;
    if (agg >= num_aggs) return;

    int row_start = agg_row_offsets[agg];
    int row_end   = agg_row_offsets[agg + 1];
    int agg_size  = row_end - row_start;

    // M lives in global memory; each block has its own slice
    ValueType *M = scratch + (size_t)agg * max_agg_rows * null_dim;

    // sh_norm and sh_dot are tiny — keep in shared memory
    extern __shared__ char smem_raw[];
    ValueType *sh_norm = reinterpret_cast<ValueType *>(smem_raw);
    ValueType *sh_dot  = sh_norm + null_dim;

    int tid = threadIdx.x;

    // ---- Load B rows into global scratch M ----
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            M[local_row * null_dim + k] = B[k * num_rows + global_row];
    }
    __threadfence();
    __syncthreads();

    // ---- Modified Gram-Schmidt QR (same algorithm, M in global memory) ----
    for (int k = 0; k < null_dim; ++k)
    {
        ValueType partial = static_cast<ValueType>(0);
        for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
        {
            ValueType v = M[local_row * null_dim + k];
            partial += v * v;
        }

        if (tid == 0) sh_norm[k] = static_cast<ValueType>(0);
        __syncthreads();
        atomicAdd(&sh_norm[k], partial);
        __syncthreads();

        ValueType norm = sqrt(sh_norm[k]);
        if (norm < static_cast<ValueType>(1e-14)) norm = static_cast<ValueType>(1);

        if (tid == 0)
            R_out[agg * null_dim * null_dim + k * null_dim + k] = norm;

        ValueType inv_norm = static_cast<ValueType>(1) / norm;
        for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
            M[local_row * null_dim + k] *= inv_norm;
        __syncthreads();

        for (int j = k + 1; j < null_dim; ++j)
        {
            ValueType pdot = static_cast<ValueType>(0);
            for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
                pdot += M[local_row * null_dim + k] * M[local_row * null_dim + j];

            if (tid == 0) sh_dot[j] = static_cast<ValueType>(0);
            __syncthreads();
            atomicAdd(&sh_dot[j], pdot);
            __syncthreads();

            if (tid == 0)
                R_out[agg * null_dim * null_dim + j * null_dim + k] = sh_dot[j];

            ValueType dot = sh_dot[j];
            for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
                M[local_row * null_dim + j] -= dot * M[local_row * null_dim + k];
            __syncthreads();
        }
    }

    // ---- Write Q to P_tent_vals ----
    for (int local_row = tid; local_row < agg_size; local_row += blockDim.x)
    {
        int global_row = agg_rows[row_start + local_row];
        for (int k = 0; k < null_dim; ++k)
            P_tent_vals[global_row * null_dim + k] = M[local_row * null_dim + k];
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
    // Count rows per aggregate.
    // Skip excluded rows (aggregates[i] == -1, Approach A).
    thrust::device_ptr<const int> agg_ptr(aggregates);
    thrust::device_ptr<int> offsets_ptr(agg_row_offsets);
    thrust::device_ptr<int> rows_ptr(agg_rows);

    // Zero offsets
    thrust::fill(offsets_ptr, offsets_ptr + num_aggs + 1, 0);

    // Count: for each valid row, atomicAdd counts[agg].
    thrust::device_vector<int> counts(num_aggs, 0);
    const int *agg_raw_ptr = aggregates;
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_rows),
        [agg_raw_ptr, counts_raw = thrust::raw_pointer_cast(counts.data())] __device__ (int i) {
            int agg_id = agg_raw_ptr[i];
            if (agg_id >= 0) atomicAdd(counts_raw + agg_id, 1);
        });

    // Exclusive scan to get offsets.
    thrust::exclusive_scan(counts.begin(), counts.end(), offsets_ptr);
    // Set last entry = sum of all counts (total valid rows, not num_rows).
    {
        int total_valid;
        thrust::device_ptr<int> last_count = offsets_ptr + num_aggs - 1;
        thrust::device_ptr<int> last_cnt_vec(thrust::raw_pointer_cast(counts.data()) + num_aggs - 1);
        // last_offset = offsets[num_aggs-1] + counts[num_aggs-1]
        int last_off_h, last_cnt_h;
        cudaMemcpy(&last_off_h, agg_row_offsets + num_aggs - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_cnt_h, thrust::raw_pointer_cast(counts.data()) + num_aggs - 1, sizeof(int), cudaMemcpyDeviceToHost);
        total_valid = last_off_h + last_cnt_h;
        cudaMemcpy(agg_row_offsets + num_aggs, &total_valid, sizeof(int), cudaMemcpyHostToDevice);
    }

    // Fill agg_rows: for each valid fine row i, place i at position
    // offsets[agg]+local_idx using atomic counters.
    thrust::device_vector<int> cursor(num_aggs);
    thrust::copy(offsets_ptr, offsets_ptr + num_aggs, cursor.begin());

    int *cursor_raw = thrust::raw_pointer_cast(cursor.data());
    int *rows_raw   = thrust::raw_pointer_cast(rows_ptr);
    const int *agg_raw = aggregates;

    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_rows),
        [agg_raw, cursor_raw, rows_raw] __device__ (int i) {
            int a = agg_raw[i];
            if (a >= 0) {
                int pos = atomicAdd(cursor_raw + a, 1);
                rows_raw[pos] = i;
            }
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

    // Compute max aggregate size (in DOF rows) across all aggregates.
    // agg_row_offsets has num_aggs+1 entries on device.
    // max_agg_size = max over a of (agg_row_offsets[a+1] - agg_row_offsets[a]).
    int max_agg_size = 0;
    {
        thrust::device_ptr<const int> offs(agg_row_offsets);
        // Compute sizes[a] = offs[a+1] - offs[a] via transform, then reduce.
        thrust::device_vector<int> sizes(num_aggs);
        thrust::transform(offs, offs + num_aggs,
                          offs + 1,
                          sizes.begin(),
                          [] __device__ (int a, int b) { return b - a; });
        max_agg_size = thrust::reduce(sizes.begin(), sizes.end(), 0,
                                      thrust::maximum<int>());
    }

    if (max_agg_size == 0) return;

    // Dynamic shared memory size for fast path:
    //   M[max_agg_size * null_dim] + sh_norm[null_dim] + sh_dot[null_dim]
    size_t smem_bytes = static_cast<size_t>(max_agg_size * null_dim + 2 * null_dim)
                        * sizeof(ValueType);

    // Shared memory for gmem fallback path: only sh_norm + sh_dot (tiny)
    size_t smem_bytes_gmem = static_cast<size_t>(2 * null_dim) * sizeof(ValueType);

    int threads = 256;
    int blocks  = num_aggs;

    // Query device shared memory limit
    size_t smem_limit = 0;
    {
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        smem_limit = prop.sharedMemPerBlock;
    }

    if (smem_bytes <= smem_limit)
    {
        // Fast path: M in shared memory
        batched_qr_kernel<ValueType><<<blocks, threads, smem_bytes>>>(
            num_aggs, null_dim, max_agg_size,
            agg_row_offsets, agg_rows,
            B, num_rows,
            P_tent_vals, R_out);
        cudaCheckError();
    }
    else
    {
        // Fallback path: M in global memory scratch buffer
        fprintf(stderr,
                "[batched_qr] smem needed=%zu > limit=%zu; "
                "using global-memory fallback (max_agg_size=%d, null_dim=%d)\n",
                smem_bytes, smem_limit, max_agg_size, null_dim);

        size_t scratch_elems = (size_t)num_aggs * max_agg_size * null_dim;
        thrust::device_vector<ValueType> scratch(scratch_elems, static_cast<ValueType>(0));
        ValueType *scratch_ptr = thrust::raw_pointer_cast(scratch.data());

        batched_qr_kernel_gmem<ValueType><<<blocks, threads, smem_bytes_gmem>>>(
            num_aggs, null_dim, max_agg_size,
            agg_row_offsets, agg_rows,
            B, num_rows,
            P_tent_vals, R_out,
            scratch_ptr);
        cudaCheckError();
    }
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
