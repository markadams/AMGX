# Aggregate Size Capping Plan — Close AMGx/PETSc Iteration Gap

## Problem Summary

AMGx MIS-2 produces aggregates averaging ~9.5 nodes (1M/105K = 9.5) while PETSc's MIS-2 produces ~6.8 nodes (1M/147K = 6.8) and PETSc's square graph produces ~7.2 nodes (1M/140K = 7.2). This leads to 27 CG iterations in AMGx vs 15-17 in PETSc.

## Root Cause

AMGx's Galerkin MIS-2 approach:
1. Pass 0: MIS-1 on original graph (5 nnz/row) → ~200K roots (coarsening ~5:1)
2. Pass 1: MIS-1 on Galerkin coarse graph (~13 nnz/row) → ~105K final aggregates

The denser coarse graph in pass 1 means each MIS root covers more coarse nodes, which map back to even more fine nodes. The composition amplifies aggregate sizes.

PETSc's greedy sequential MIS assigns only distance-1 neighbors to each root, naturally limiting aggregate size to degree+1 of the graph.

## Solution: Aggregate Size Capping with Reassignment

After MIS-2 produces the composed aggregates, add a post-processing step that:
1. Counts the size of each aggregate
2. For aggregates exceeding `max_aggregate_size`, reassigns excess nodes to neighboring smaller aggregates using the original fine graph connectivity

This is analogous to PETSc's `fixAggregatesWithSquare` which uses the original graph to reassign nodes to closer roots.

## Implementation Steps

### Step 1: Register `max_aggregate_size` parameter

**File**: `src/core.cu` (line 478, after `aggressive_levels`)

```cpp
AMG_Config::registerParameter<int>("max_aggregate_size", "for selector=MIS: maximum aggregate size; oversized aggregates are split (0=no limit) <0>", 0);
```

### Step 2: Add member variable to header

**File**: `include/aggregation/selectors/mis_selector.h` (line 41, after `m_call_count`)

```cpp
int m_max_aggregate_size;  // Maximum aggregate size (0=no limit)
```

### Step 3: Read parameter in constructor

**File**: `src/aggregation/selectors/mis_selector.cu` (line 454, after `m_call_count = 0;`)

```cpp
m_max_aggregate_size = cfg.AMG_Config::template getParameter<int>("max_aggregate_size", cfg_scope);
```

### Step 4: Add aggregate splitting kernel

**File**: `src/aggregation/selectors/mis_selector.cu` (after `make_singletons_kernel`, around line 440)

Add a new kernel `split_oversized_aggregates_kernel` that:
- Each thread handles one node
- Computes aggregate sizes using atomicAdd into a shared/global counter
- For nodes in oversized aggregates (size > max_size): finds the best neighboring aggregate that is under the size limit and reassigns

The algorithm:
1. Count aggregate sizes (parallel histogram)
2. For each node in an oversized aggregate, check if any neighbor belongs to a smaller aggregate
3. Reassign the node to the smallest neighboring aggregate (preferring the one with strongest edge weight)
4. Iterate until no oversized aggregates remain (or max iterations reached)

### Step 5: Call splitting after final renumber

**File**: `src/aggregation/selectors/mis_selector.cu` (after line 907, before the final diagnostic print)

Insert the splitting logic between the final `renumberAndCountAggregates` and the diagnostic print.

### Step 6: Update test config

**File**: `examples/test_multilevel_cheby.c` (line 75)

Add `"max_aggregate_size": 8,` to the AMG config.

## Detailed Kernel Design

```cuda
// Kernel: count_aggregate_sizes_kernel
// Counts how many nodes belong to each aggregate
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

// Kernel: reassign_from_oversized_kernel
// For nodes in oversized aggregates, find a neighboring aggregate under the limit
// and reassign. Uses edge weights to pick the best neighbor.
template <typename IndexType>
__global__
void reassign_from_oversized_kernel(
    const IndexType *row_offsets,
    const IndexType *col_indices,
    const float     *edge_weights,
    IndexType       *aggregates,
    const int       *agg_sizes,
    const int        max_agg_size,
    const int        num_rows,
    int             *num_reassigned)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows) {
        int my_agg = aggregates[tid];
        // Only process nodes in oversized aggregates
        // Don't move the root node (aggregates[tid] == tid after renumber? No - roots are renumbered)
        if (my_agg >= 0 && agg_sizes[my_agg] > max_agg_size) {
            float best_weight = -1.0f;
            int   best_agg    = -1;

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++) {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) continue;

                int jagg = aggregates[jcol];
                // Prefer neighbors in smaller aggregates
                if (jagg >= 0 && jagg != my_agg && agg_sizes[jagg] < max_agg_size) {
                    float w = edge_weights[j];
                    if (w > best_weight) {
                        best_weight = w;
                        best_agg = jagg;
                    }
                }
            }

            if (best_agg >= 0) {
                aggregates[tid] = best_agg;
                atomicAdd(num_reassigned, 1);
                // Note: agg_sizes becomes stale but we recompute next iteration
            }
        }
        tid += gridDim.x * blockDim.x;
    }
}
```

## Host-side Logic

```cpp
// After renumberAndCountAggregates produces final aggregates:
if (this->m_max_aggregate_size > 0 && effective_k > 1)
{
    // Iterative splitting loop
    IVector_d agg_sizes(num_aggregates, 0);
    IVector_d d_reassigned(1);
    
    for (int split_iter = 0; split_iter < 10; split_iter++)
    {
        // Count aggregate sizes
        thrust_wrapper::fill<AMGX_device>(agg_sizes.begin(), agg_sizes.end(), 0);
        count_aggregate_sizes_kernel<<<num_blocks, threads_per_block, 0, str>>>(
            aggregates.raw(), agg_sizes.raw(), num_block_rows);
        cudaCheckError();
        
        // Reassign nodes from oversized aggregates
        cudaMemsetAsync(d_reassigned.raw(), 0, sizeof(int), str);
        reassign_from_oversized_kernel<<<num_blocks, threads_per_block, 0, str>>>(
            A.row_offsets.raw(), A.col_indices.raw(),
            edge_weights_orig.raw(), aggregates.raw(),
            agg_sizes.raw(), this->m_max_aggregate_size,
            num_block_rows, d_reassigned.raw());
        cudaCheckError();
        
        int h_reassigned = 0;
        cudaMemcpyAsync(&h_reassigned, d_reassigned.raw(), sizeof(int),
                        cudaMemcpyDeviceToHost, str);
        cudaStreamSynchronize(str);
        
        if (h_reassigned == 0) break;
        
        amgx_printf("[MIS-k] Split iter %d: reassigned %d nodes\n",
                    split_iter, h_reassigned);
    }
    
    // Re-renumber after splitting (some aggregates may now be empty)
    this->renumberAndCountAggregates(aggregates, aggregates_global,
                                     num_block_rows, num_aggregates);
    
    amgx_printf("[MIS-k] After splitting (max_size=%d): %d aggregates "
                "(avg size %.1f)\n",
                this->m_max_aggregate_size, num_aggregates,
                (float)num_block_rows / (float)num_aggregates);
}
```

## Config Change for Testing

In `examples/test_multilevel_cheby.c`, add to the AMG preconditioner config:
```json
"max_aggregate_size": 8
```

Target: ~140K aggregates (avg 7.1 nodes) to match PETSc's square graph result.

## Expected Outcome

- Aggregate count: ~105K → ~140K (matching PETSc)
- Grid complexity: ~1.12 → ~1.17 (matching PETSc)
- Operator complexity: ~1.34 → ~1.43 (matching PETSc)
- CG iterations: 27 → ~15-17 (matching PETSc)

## Files to Modify

1. `src/core.cu` — register `max_aggregate_size` parameter
2. `include/aggregation/selectors/mis_selector.h` — add `m_max_aggregate_size` member
3. `src/aggregation/selectors/mis_selector.cu` — add splitting kernels and logic
4. `examples/test_multilevel_cheby.c` — add `max_aggregate_size` to config

## Build and Test

```bash
# Local build
cd /Users/markadams/Codes/amgx/build_perlmutter
make -j16 amgxsh

# Sync to Perlmutter
cd /Users/markadams/Codes/amgx
git add -A && git commit -m "Add aggregate size capping" && git push

# On Perlmutter
cd ~/amgx-sa && git pull && cd build_perlmutter && make -j16 amgxsh
# Rebuild test binary
./test_multilevel_cheby poisson2d_1000.mtx
```
