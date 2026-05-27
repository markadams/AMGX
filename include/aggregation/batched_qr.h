// SPDX-FileCopyrightText: 2011 - 2025 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace amgx
{
namespace aggregation
{

/**
 * @brief Batched per-aggregate Modified Gram-Schmidt QR factorization.
 *
 * For each aggregate a, gathers the rows of the near-null space matrix B
 * that belong to aggregate a into a local dense matrix M of size
 * (agg_size_a * block_size) x null_dim, then computes M = Q * R via
 * Modified Gram-Schmidt.
 *
 * Outputs:
 *   - Q values stored in P_tent_vals (same ordering as the P_tent sparsity
 *     pattern: for each fine row i, null_dim consecutive values corresponding
 *     to the null_dim columns of Q for that row)
 *   - R stored in R_out (column-major, null_dim x null_dim per aggregate,
 *     stored as num_aggs * null_dim * null_dim contiguous doubles)
 *
 * Layout conventions:
 *   B is column-major: null vector k at B + k * num_rows, length num_rows.
 *   agg_row_offsets[a] .. agg_row_offsets[a+1]-1 gives the fine rows in agg a.
 *   agg_rows[agg_row_offsets[a] .. agg_row_offsets[a+1]-1] are the fine row ids.
 *   P_tent_vals[i * null_dim .. (i+1)*null_dim - 1] = Q row i (null_dim values).
 *   R_out[a * null_dim * null_dim .. (a+1)*null_dim*null_dim - 1] = R for agg a,
 *     column-major null_dim x null_dim.
 *
 * @param num_aggs        Number of aggregates.
 * @param null_dim        Number of near-null space vectors (columns of B).
 * @param num_rows        Total number of fine-level rows (= num_nodes * block_size).
 * @param agg_row_offsets Device array of length num_aggs+1 (CSR row offsets into agg_rows).
 * @param agg_rows        Device array of fine row indices sorted by aggregate.
 * @param B               Device array: near-null space, column-major, length num_rows*null_dim.
 * @param P_tent_vals     Device output: Q values, length num_rows * null_dim.
 * @param R_out           Device output: R factors, length num_aggs * null_dim * null_dim.
 */
template <typename ValueType>
void batched_qr(int num_aggs,
                int null_dim,
                int num_rows,
                const int    *agg_row_offsets,
                const int    *agg_rows,
                const ValueType *B,
                ValueType    *P_tent_vals,
                ValueType    *R_out);

/**
 * @brief Build the CSR row-list representation of aggregates.
 *
 * Given aggregates[i] = aggregate index for fine row i, produces:
 *   agg_row_offsets[a] = start index in agg_rows for aggregate a
 *   agg_rows[k]        = fine row index
 *
 * @param num_rows        Number of fine rows.
 * @param num_aggs        Number of aggregates.
 * @param aggregates      Device array of length num_rows: aggregates[i] = agg id.
 * @param agg_row_offsets Device output array of length num_aggs+1.
 * @param agg_rows        Device output array of length num_rows.
 */
void build_agg_row_lists(int num_rows,
                         int num_aggs,
                         const int *aggregates,
                         int *agg_row_offsets,
                         int *agg_rows);

} // namespace aggregation
} // namespace amgx
