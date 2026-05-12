# Quality-Aware Aggregate Refinement Plan

## Algorithm: Iterative Quality-Aware Refinement

### Overview

After MIS-2 (Galerkin loop) produces initial aggregates, apply iterative refinement that:
1. Ignores weak edges (won't assign across weak connections)
2. Promotes aggregate size uniformity (prefers smaller target aggregates)
3. Is race-free and deadlock-free (each node independently decides)

### Algorithm Steps

```
Step 1: Run MIS-2 as before (Galerkin loop with MIS-1 per pass)
        → Produces initial aggregates (currently avg ~9.5 nodes)

Step 2: Compute aggregate sizes (parallel histogram via atomicAdd)

Step 3: Iterative refinement loop (max 10 iterations):
    For each node i in an oversized aggregate (agg_sizes[agg[i]] > max_size):
        Compute threshold = max_edge_weight_in_row(i) * alpha  (e.g., alpha=0.1)
        For each neighbor j of node i in the ORIGINAL fine graph:
            Skip if edge_weight(i,j) < threshold  (weak edge)
            Skip if agg[j] == agg[i]  (same aggregate)
            Compute score = edge_weight(i,j) / agg_sizes[agg[j]]
        If best_score found and target aggregate is smaller:
            Reassign: agg[i] = agg[best_j]
    
    Recompute aggregate sizes
    If no reassignments made → break

Step 4: Re-renumber aggregates (fill gaps from emptied aggregates)
```

### Key Properties

- **Race-free**: Each node independently decides which aggregate to join.
  No two threads write to the same memory location.
- **Deadlock-free**: No mutual dependencies. Single parallel pass per iteration.
- **Quality-aware**: Score = edge_weight / agg_size rewards:
  - Strong connections (numerator)
  - Small target aggregates (denominator)
- **Weak-edge filtering**: Threshold prevents creating bad aggregates
  across weak connections that don't represent strong physical coupling.
- **Convergence**: Each iteration reduces max aggregate size monotonically
  (nodes only move FROM oversized aggregates TO smaller ones).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_aggregate_size` | 0 (disabled) | Trigger threshold for refinement. Set to 8 for ~7 avg. |
| `alpha` (internal) | 0.1 | Weak edge threshold as fraction of max edge weight per row |
| Max refinement iters | 10 | Safety limit on refinement iterations |

### GPU Kernel Design

```cuda
// Kernel 1: count_aggregate_sizes_kernel
__global__
void count_aggregate_sizes_kernel(const int *aggregates, int *agg_sizes,
                                   const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows) {
        int agg = aggregates[tid];
        if (agg >= 0) atomicAdd(&agg_sizes[agg], 1);
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel 2: compute_max_edge_weight_per_row_kernel
// (needed for threshold computation)
template <typename IndexType>
__global__
void compute_max_edge_weight_kernel(const IndexType *row_offsets,
                                     const float *edge_weights,
                                     float *max_edge_weight,
                                     const int num_rows)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows) {
        float max_w = 0.0f;
        int jmin = row_offsets[tid];
        int jmax = row_offsets[tid + 1];
        for (int j = jmin; j < jmax; j++) {
            float w = edge_weights[j];
            if (w > max_w) max_w = w;
        }
        max_edge_weight[tid] = max_w;
        tid += gridDim.x * blockDim.x;
    }
}

// Kernel 3: refine_aggregates_kernel
// For nodes in oversized aggregates, find better assignment
template <typename IndexType>
__global__
void refine_aggregates_kernel(
    const IndexType *row_offsets,
    const IndexType *col_indices,
    const float     *edge_weights,
    const float     *max_edge_weight,  // per-row max edge weight
    IndexType       *aggregates,
    const int       *agg_sizes,
    const int        max_agg_size,
    const float      alpha,            // weak edge threshold fraction
    const int        num_rows,
    int             *num_reassigned)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows) {
        int my_agg = aggregates[tid];
        // Only process nodes in oversized aggregates
        if (my_agg >= 0 && agg_sizes[my_agg] > max_agg_size) {
            float threshold = max_edge_weight[tid] * alpha;
            float best_score = -1.0f;
            int   best_agg   = -1;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++) {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) continue;

                float w = edge_weights[j];
                if (w < threshold) continue;  // skip weak edges

                int jagg = aggregates[jcol];
                if (jagg >= 0 && jagg != my_agg && agg_sizes[jagg] < max_agg_size) {
                    // Score: edge_weight / target_agg_size
                    float score = w / (float)(agg_sizes[jagg] + 1);
                    if (score > best_score || (score == best_score && jcol > tid)) {
                        best_score = score;
                        best_agg = jagg;
                    }
                }
            }

            if (best_agg >= 0) {
                aggregates[tid] = best_agg;
                atomicAdd(num_reassigned, 1);
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}
```

### Host-Side Logic

```cpp
// After renumberAndCountAggregates produces final aggregates:
if (this->m_max_aggregate_size > 0 && effective_k > 1)
{
    const float alpha = 0.1f;  // weak edge threshold
    
    // Compute max edge weight per row (for threshold)
    FVector_d max_ew_per_row(num_block_rows);
    compute_max_edge_weight_kernel<<<num_blocks, tpb, 0, str>>>(
        A.row_offsets.raw(), edge_weights_orig.raw(),
        max_ew_per_row.raw(), num_block_rows);
    cudaCheckError();
    
    IVector_d agg_sizes(num_aggregates, 0);
    IVector_d d_reassigned(1);
    
    for (int refine_iter = 0; refine_iter < 10; refine_iter++)
    {
        // Count aggregate sizes
        thrust_wrapper::fill<AMGX_device>(agg_sizes.begin(), agg_sizes.end(), 0);
        count_aggregate_sizes_kernel<<<num_blocks, tpb, 0, str>>>(
            aggregates.raw(), agg_sizes.raw(), num_block_rows);
        cudaCheckError();
        
        // Reassign nodes from oversized aggregates
        cudaMemsetAsync(d_reassigned.raw(), 0, sizeof(int), str);
        refine_aggregates_kernel<<<num_blocks, tpb, 0, str>>>(
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
    
    // Re-renumber after refinement
    this->renumberAndCountAggregates(aggregates, aggregates_global,
                                     num_block_rows, num_aggregates);
    
    amgx_printf("[MIS-k] After refinement (max_size=%d): %d aggregates "
                "(avg size %.1f)\n",
                this->m_max_aggregate_size, num_aggregates,
                (float)num_block_rows / (float)num_aggregates);
}
```

### Files to Modify

1. `src/core.cu` line 478 — Register `max_aggregate_size` parameter
2. `include/aggregation/selectors/mis_selector.h` line 41 — Add `m_max_aggregate_size` member
3. `src/aggregation/selectors/mis_selector.cu` — Add 3 kernels + host logic
4. `examples/test_multilevel_cheby.c` line 75 — Add `"max_aggregate_size": 8`

### Expected Outcome

With `max_aggregate_size=8`:
- Aggregates: 105K → ~140K (avg ~7.1 nodes)
- Grid complexity: 1.12 → ~1.17
- Operator complexity: 1.34 → ~1.43
- CG iterations: 27 → ~15-17

### Convergence Guarantee

The algorithm converges because:
1. Nodes only move FROM oversized aggregates (size > max) TO undersized ones (size < max)
2. Each reassignment reduces the source aggregate size by 1 and increases target by 1
3. Eventually all aggregates are ≤ max_size, or no valid moves remain
4. The 10-iteration cap prevents pathological cases
