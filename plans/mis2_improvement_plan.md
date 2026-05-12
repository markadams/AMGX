# MIS-2 Aggregation Quality Improvement — Detailed Implementation Plan

## Goal

Close the iteration count gap between AMGx (27 CG iterations) and PETSc GAMG (15-17 CG iterations) on the 1000×1000 2D Poisson problem (1M DOFs).

## Current Baseline

| Solver | Method | L1 rows (aggregates) | CG Iters | Avg Agg Size |
|--------|--------|---------------------:|:---------:|:------------:|
| PETSc GAMG | Square graph | 139,863 | **15** | 7.2 |
| PETSc GAMG | MIS-2 | 147,053 | **17** | 6.8 |
| **AMGx** | **MIS-2 Galerkin** | **105,204** | **27** | **9.5** |

## Two Approaches

- **Approach A**: Quality-aware aggregate refinement (post-processing on existing MIS-2)
- **Approach B**: Implicit square graph MIS-2 (new algorithm, no Galerkin product)

Both will be implemented. Approach A is additive and low-risk. Approach B is a new code path.

---

## Edge Weight Details

### Formula (from common_selector.h)

**weight_formula=0** (default):
```
w(i,j) = 0.5 * (|a_ij| + |a_ji|) / max(|a_ii|, |a_jj|)
```

**weight_formula=1**:
```
w(i,j) = -0.5 * (a_ij/a_ii + a_ji/a_jj)
```

### Key Properties
- **Value-symmetric by construction**: `w(i,j) == w(j,i)` always (formula averages both directions)
- **Structural asymmetry**: If `a_ji` doesn't exist in CSR, `foundk=false` → `w(i,j) = 0`. Edge effectively ignored.
- **MPI limitation**: Current code skips `j >= num_owned` → halo edge weights are 0. This makes the algorithm behave differently in parallel. **Needs attention for MPI** (see MPI notes below).

### Correctness Analysis for Refinement

**Value asymmetry (a_ij ≠ a_ji)**: Not a problem. Edge weights are symmetric by formula.

**Structural asymmetry (a_ij exists, a_ji doesn't)**: Not a failure. Such edges get weight 0 and are ignored by both the existing MIS and the refinement. Aggregation quality may be suboptimal but no crash/deadlock.

**Per-row threshold creates asymmetric visibility**: Even with symmetric weights, `max_edge_weight[i]` may differ from `max_edge_weight[j]`, so the threshold `alpha * max_ew` is different per row. This means node `i` might "see" edge (i,j) but node `j` might not "see" edge (j,i). **This cannot cause failure** — the refinement is a unilateral decision by each node (no acceptance needed). It may produce slightly different aggregates than if the threshold were symmetric, but this is acceptable.

**Conclusion**: The refinement algorithm is safe for any matrix. It cannot deadlock, race, or crash. For structurally asymmetric matrices, we add a diagnostic warning suggesting symmetrization for better quality.

---

## MPI Parallel Notes

### Known Issues (to address in MPI parallel plan)

1. **Edge weights for halo columns are 0**: The `computeEdgeWeightsBlockDiaCsr_V2` kernel skips `j >= num_owned`. For MPI correctness, edge weights need to be exchanged so boundary nodes have correct weights for halo neighbors.

2. **Node weights use local IDs**: Currently `hash(local_tid)` which is partition-dependent. For deterministic (partition-independent) results, should use `hash(global_id)`.

3. **Determinism guarantee**: If node weights use `hash(global_id)` AND edge weights are communicated, the algorithm produces identical aggregates regardless of partitioning. This is a desirable property for debugging and reproducibility.

4. **Refinement in MPI**: The refinement kernel currently only considers `jcol < num_rows` (owned nodes). For MPI, it should consider halo neighbors too (with communicated edge weights and aggregate IDs). This requires `exchange_halo` of the aggregates array during refinement iterations.

**These MPI issues are deferred to the MPI parallel plan. The current implementation targets single-GPU correctness.**

---

## Approach A: Quality-Aware Aggregate Refinement

### Concept

After the existing Galerkin MIS-2 produces aggregates, iteratively move nodes from oversized aggregates to smaller neighboring aggregates. The reassignment uses a quality score that:
- Rewards strong edge connections (numerator)
- Rewards small target aggregates (denominator)
- Ignores weak edges below a user-configurable threshold

### Development Phases

---

#### Phase A1: Register Parameters and Add Member Variables

**Model**: Sonnet (mechanical code change, low complexity)

**Status**: Not started

**Changes**:

1. **`src/core.cu`** — Add after line 477 (after `aggressive_levels` registration):
```cpp
AMG_Config::registerParameter<int>("max_aggregate_size",
    "for selector=MIS: max aggregate size for quality refinement (0=no limit) <0>", 0);
AMG_Config::registerParameter<double>("refine_threshold",
    "for selector=MIS: weak edge threshold as fraction of max edge weight per row for refinement <0.1>", 0.1, 0.0, 1.0);
```

2. **`include/aggregation/selectors/mis_selector.h`** — Add after line 41 (after `m_call_count`):
```cpp
int m_max_aggregate_size;    // Maximum aggregate size (0=no limit, triggers refinement)
double m_refine_threshold;   // Weak edge threshold fraction for refinement (0.0-1.0)
```

3. **`src/aggregation/selectors/mis_selector.cu`** — Add in constructor after line 454 (after `m_call_count = 0;`):
```cpp
m_max_aggregate_size = cfg.AMG_Config::template getParameter<int>("max_aggregate_size", cfg_scope);
m_refine_threshold = cfg.AMG_Config::template getParameter<double>("refine_threshold", cfg_scope);
```

**Verification**:
- Build compiles without errors
- Run existing test: `./test_multilevel_cheby poisson2d_1000.mtx` — identical results (parameter defaults to 0 = disabled)

---

#### Phase A2: Add Aggregate Size Counting Kernel

**Model**: Sonnet (simple kernel, follows existing patterns)

**Status**: Not started

**Changes**:

Add in `src/aggregation/selectors/mis_selector.cu` after `make_singletons_kernel` (around line 440):

```cuda
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
```

**Verification**: Build compiles. No functional change (kernel not called yet).

---

#### Phase A3: Add Max Edge Weight Per Row Kernel

**Model**: Sonnet (simple kernel)

**Status**: Not started

**Changes**:

Add after `count_aggregate_sizes_kernel`:

```cuda
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
```

**Verification**: Build compiles. No functional change.

---

#### Phase A4: Add Refinement Kernel

**Model**: Sonnet (moderate kernel, follows existing patterns closely)

**Status**: Not started

**Changes**:

Add after `compute_max_edge_weight_kernel`:

```cuda
// -----------------------------------------------------------------------
// Kernel: refine_aggregates_kernel
//
// For nodes in oversized aggregates (size > max_agg_size), find a better
// aggregate assignment using quality-aware scoring:
//   score = edge_weight(i,j) / (agg_sizes[agg[j]] + 1)
//
// Skips weak edges (below alpha * max_edge_weight_for_row).
// Only moves nodes TO aggregates that are below the size limit.
// Race-free: each node independently decides its new aggregate.
//
// The algorithm cannot fail for asymmetric matrices:
// - Value asymmetry: edge weights are symmetric by formula
// - Structural asymmetry: missing edges have weight 0, ignored
// - Per-row threshold asymmetry: unilateral decision, no acceptance needed
//
// num_reassigned is incremented atomically for convergence detection.
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

            int jmin = row_offsets[tid];
            int jmax = row_offsets[tid + 1];

            for (int j = jmin; j < jmax; j++)
            {
                int jcol = col_indices[j];
                if (jcol == tid || jcol >= num_rows) continue;

                float w = edge_weights[j];
                if (w < threshold) continue;  // skip weak edges

                int jagg = aggregates[jcol];
                if (jagg >= 0 && jagg != my_agg && agg_sizes[jagg] < max_agg_size)
                {
                    // Quality score: strong edge to small aggregate
                    float score = w / (float)(agg_sizes[jagg] + 1);
                    if (score > best_score)
                    {
                        best_score = score;
                        best_agg = jagg;
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
```

**Verification**: Build compiles. No functional change.

---

#### Phase A5: Add Host-Side Refinement Logic

**Model**: Opus (needs to understand the flow and insert code at the right location)

**Status**: Not started

**Changes**:

In `src/aggregation/selectors/mis_selector.cu`, after the final `renumberAndCountAggregates` call (line 907) and before the final diagnostic print (line 909), insert:

```cpp
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
    // ------------------------------------------------------------------
    if (this->m_max_aggregate_size > 0 && effective_k > 1)
    {
        const float alpha = (float)this->m_refine_threshold;
        const int max_refine_iters = 10;
        const int num_blocks_fine = std::min(AMGX_GRID_MAX_SIZE,
                                             (num_block_rows - 1) / threads_per_block + 1);

        // Diagnostic: check for structural asymmetry (optional warning)
        // TODO: Add structural symmetry check and warn if asymmetric

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
```

**Verification**:
- Build compiles
- Run with `max_aggregate_size=0` (default): identical to baseline (27 iters)
- Run with `max_aggregate_size=8`: should see refinement messages and reduced iteration count
- Expected: aggregates increase from ~105K to ~130-150K, iterations decrease from 27 to ~15-20

---

#### Phase A6: Update Test Config

**Model**: Sonnet (trivial change)

**Status**: Not started

**Changes**:

In `examples/test_multilevel_cheby.c`, line 75, add after `"aggressive_levels": 1,`:
```c
"      \"max_aggregate_size\": 8,"
"      \"refine_threshold\": 0.1,"
```

**Verification**:
- Rebuild test binary on Perlmutter
- Run: `./test_multilevel_cheby poisson2d_1000.mtx`
- Compare iteration count with baseline (27) and PETSc (15-17)

---

### Approach A Summary

| Phase | Description | Model | Status |
|:-----:|-------------|:-----:|:------:|
| A1 | Register parameters + member variables | Sonnet | Not started |
| A2 | Add count_aggregate_sizes kernel | Sonnet | Not started |
| A3 | Add compute_max_edge_weight kernel | Sonnet | Not started |
| A4 | Add refine_aggregates kernel | Sonnet | Not started |
| A5 | Add host-side refinement logic | Opus | Not started |
| A6 | Update test config | Sonnet | Not started |

---

## Approach B: Implicit Square Graph MIS-2

### Concept

Instead of the Galerkin loop (MIS-1 → coarsen graph → MIS-1 → compose), use a single MIS-2 kernel that checks distance-2 neighbors directly on the original graph. Then assign aggregates using only distance-1 connectivity, which naturally limits aggregate size.

This is equivalent to running MIS-1 on the square graph A^T*A, but without explicitly forming A^T*A.

### Key Algorithmic Details

**MIS-2 Selection (Phase 1)**:
- A node becomes a ROOT if it has the highest weight among ALL undecided nodes within distance 2
- A node becomes NOT_ROOT if ANY node within distance 2 is already a ROOT
- This is the existing `find_mis_k2_kernel` but needs MPI support (halo exchange)

**Distance-1 Assignment (Phase 2)**:
- Each NOT_ROOT node finds its strongest ROOT neighbor at distance 1 ONLY
- This is the existing `assign_aggregates_kernel` — no change needed
- Some nodes will have NO ROOT at distance 1 (because MIS-2 roots can be 2 apart)

**Propagation (Phase 3)**:
- Unassigned nodes (no ROOT within distance 1) find their strongest ASSIGNED neighbor
- This is the existing `propagate_aggregates_kernel` — no change needed
- Propagation naturally limits aggregate diameter to ~2 hops from root

**Why this produces better aggregates**:
- MIS-2 roots are spaced ~2 apart (same as Galerkin MIS-2)
- But distance-1 assignment means each root only directly grabs its immediate neighbors
- Propagation adds a few more nodes, but aggregate size is bounded by local graph degree
- No "composition amplification" from the Galerkin product

**For high-order / high-degree graphs**:
- MIS-2 with distance-2 domination on a high-degree graph produces very few roots
- Distance-1 assignment then limits each aggregate to ~degree nodes
- O(degree^2) cost per node per MIS iteration is acceptable (replaces Galerkin product)

### MPI Considerations for Approach B

The `find_mis_k2_kernel` checks neighbors-of-neighbors. For MPI:
- Distance-1 neighbors include halo nodes (status exchanged via `exchange_halo`)
- Distance-2 neighbors (neighbors of halo nodes) are NOT accessible — we don't have row_offsets for halo nodes
- **Limitation**: At partition boundaries, the distance-2 check is incomplete (only through owned intermediate nodes)
- **Mitigation**: Halo exchange of status after each iteration propagates ROOT decisions. Over multiple iterations, the MIS converges correctly. Worst case: slightly non-maximal IS at boundaries (a few extra roots), which is fine for aggregation.
- **Note**: This is the same limitation as the Galerkin approach (which also has incomplete halo information)

### Development Phases

---

#### Phase B1: Register Algorithm Selection Parameter

**Model**: Sonnet (mechanical)

**Status**: Not started

**Changes**:

1. **`src/core.cu`** — Add after `refine_threshold` registration:
```cpp
AMG_Config::registerParameter<int>("mis2_algorithm",
    "for selector=MIS with mis_k=2: 0=Galerkin loop, 1=implicit square graph <0>", 0);
```

2. **`include/aggregation/selectors/mis_selector.h`** — Add after `m_refine_threshold`:
```cpp
int m_mis2_algorithm;  // 0=Galerkin loop (default), 1=implicit square graph
```

3. **`src/aggregation/selectors/mis_selector.cu`** — In constructor:
```cpp
m_mis2_algorithm = cfg.AMG_Config::template getParameter<int>("mis2_algorithm", cfg_scope);
```

**Verification**: Build compiles. Default behavior unchanged.

---

#### Phase B2: Fix find_mis_k2_kernel for MPI (Add total_rows Parameter)

**Model**: Sonnet (mechanical — follow the pattern of find_mis_k1_kernel)

**Status**: Not started

**Changes**:

Update `find_mis_k2_kernel` signature to add `total_rows` parameter. Change column skip condition from `jcol >= num_rows` to `jcol >= total_rows` for distance-1 neighbors. For distance-2 expansion, only expand owned nodes (`jcol < num_rows`) since we don't have row_offsets for halo nodes.

Key changes to the existing kernel:
1. Add `const int total_rows` parameter
2. Distance-1 loop: change `jcol >= num_rows` to `jcol >= total_rows`
3. Distance-2 loop: only expand if `jcol < num_rows` (owned nodes have row_offsets)
4. Distance-2 inner loop: change `kcol >= num_rows` to `kcol >= total_rows`

```cuda
template <typename IndexType>
__global__
void find_mis_k2_kernel(const IndexType *row_offsets,
                        const IndexType *col_indices,
                        const float     *node_weights,
                        int             *status,
                        const int        num_rows,
                        const int        total_rows,  // NEW: owned + halo
                        int             *num_changed)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    while (tid < num_rows)  // only process owned nodes
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
                // Only expand owned nodes (we have row_offsets for them)
                if (jcol < num_rows)
                {
                    int kmin = row_offsets[jcol];
                    int kmax = row_offsets[jcol + 1];
                    for (int k = kmin; k < kmax && !dominated; k++)
                    {
                        int kcol = col_indices[k];
                        if (kcol == tid || kcol == jcol || kcol >= total_rows) { continue; }

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
```

**Verification**:
- Build compiles
- Single-GPU: kernel not yet called in new path (Phase B3)
- Can verify by temporarily calling it and checking root count matches expectations

---

#### Phase B3: Add Implicit Square Graph Code Path

**Model**: Opus (needs to understand control flow and insert a new branch)

**Status**: Not started

**Changes**:

In `src/aggregation/selectors/mis_selector.cu`, restructure `setAggregates_common_sqblocks` to use an if-else for the two algorithms. The implicit square graph path:

1. Runs `find_mis_k2_kernel` iteratively with halo exchange until converged
2. Uses existing `assign_aggregates_kernel` for distance-1 assignment
3. Uses existing `propagate_aggregates_kernel` for unassigned nodes
4. Calls `renumberAndCountAggregates`
5. Falls through to the quality refinement (Approach A) if enabled

The code structure becomes:
```cpp
if (this->m_mis2_algorithm == 1 && effective_k == 2)
{
    // Implicit square graph path
    // ... (Phase 1: MIS-2, Phase 2: d=1 assign, Phase 3: propagate)
}
else
{
    // Existing Galerkin loop path (unchanged)
    // ... (all existing code from line 564 to line 907)
}

// Quality refinement (applies to both paths)
if (this->m_max_aggregate_size > 0 && effective_k > 1)
{
    // ... (refinement from Approach A)
}
```

The implicit square graph path reuses:
- `assign_node_weights_kernel` (existing)
- `find_mis_k2_kernel` (updated in Phase B2)
- `assign_aggregates_kernel` (existing)
- `propagate_aggregates_kernel` (existing)
- `join_candidates_kernel` (existing)
- `make_singletons_kernel` (existing)
- `renumberAndCountAggregates` (existing)

**New code needed**: Only the orchestration logic (~80 lines) that:
1. Allocates node_weights, status_vec
2. Runs MIS-2 iteration loop with halo exchange
3. Calls assign_aggregates_kernel
4. Runs propagation loop
5. Calls renumberAndCountAggregates

**Verification**:
- Build compiles
- Run with `mis2_algorithm=0`: identical to baseline (Galerkin path)
- Run with `mis2_algorithm=1`: should see implicit MIS-2 messages
- Check aggregate count: should be different from Galerkin (likely ~130-160K)
- Check iteration count: should be lower than 27

---

#### Phase B4: Update Test Config for Approach B

**Model**: Sonnet (trivial)

**Status**: Not started

**Changes**:

Add `mis2_algorithm` to test config. Can test both by running twice with different configs, or add a second test binary.

**Verification**: Run both configurations on Perlmutter, compare results.

---

### Approach B Summary

| Phase | Description | Model | Status |
|:-----:|-------------|:-----:|:------:|
| B1 | Register algorithm parameter | Sonnet | Not started |
| B2 | Fix find_mis_k2_kernel for MPI (total_rows) | Sonnet | Not started |
| B3 | Add implicit square graph code path | Opus | Not started |
| B4 | Update test config | Sonnet | Not started |

---

## Combined Testing Matrix

After both approaches are implemented:

| Config | mis2_algorithm | max_aggregate_size | Expected Result |
|--------|:-:|:-:|---|
| Baseline | 0 | 0 | 105K aggs, 27 iters |
| Approach A only | 0 | 8 | ~140K aggs, ~17-20 iters |
| Approach B only | 1 | 0 | ~130-160K aggs, ~15-20 iters |
| Both combined | 1 | 8 | ~140K aggs, ~15-17 iters |

---

## Implementation Order

1. **Phase A1-A4** (Sonnet): Register parameters, add kernels (no functional change)
2. **Phase A5** (Opus): Add host-side refinement logic
3. **Phase A6** (Sonnet): Update test config
4. **Test Approach A on Perlmutter**
5. **Phase B1** (Sonnet): Register algorithm parameter
6. **Phase B2** (Sonnet): Fix find_mis_k2_kernel
7. **Phase B3** (Opus): Add implicit square graph code path
8. **Phase B4** (Sonnet): Update test config
9. **Test Approach B on Perlmutter**
10. **Test combined on Perlmutter**

---

## Build and Test Commands

```bash
# Sync to Perlmutter
cd /Users/markadams/Codes/amgx
git add -A && git commit -m "MIS-2 quality improvement" && git push

# On Perlmutter (SSH)
ssh -i ~/.ssh/nersc madams@perlmutter-p1.nersc.gov
cd ~/amgx-sa && git pull
cd build_perlmutter && make -j16 amgxsh

# Rebuild test binary
CUDA_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9/targets/x86_64-linux/lib
MATH_LIB=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/lib64
/opt/cray/pe/gcc/12.2.0/bin/gcc -O2 -I../include -DRAPIDJSON_DEFINED \
  ../examples/test_multilevel_cheby.c -o test_multilevel_cheby \
  ./libamgxsh.so -lrt -ldl \
  $CUDA_LIB/libcudart_static.a \
  $MATH_LIB/libcublas.so $MATH_LIB/libcusolver.so $MATH_LIB/libcusparse.so \
  $CUDA_LIB/libculibos.a $CUDA_LIB/libnvJitLink.so $MATH_LIB/libcublasLt.so \
  -lm -lpthread -ldl

# Run test
export LD_LIBRARY_PATH=$PWD:$MATH_LIB:$LD_LIBRARY_PATH
./test_multilevel_cheby poisson2d_1000.mtx
```

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Refinement doesn't converge | Max 10 iterations cap; monotonic convergence guaranteed |
| Refinement produces worse quality | Only moves from large to small; weak-edge filter prevents bad moves |
| Implicit MIS-2 too slow (O(degree^2)) | Only on aggressive levels; Galerkin path remains for high-degree |
| MPI incorrectness in find_mis_k2_kernel | Halo exchange each iter; incomplete d=2 at boundaries acceptable |
| Asymmetric matrix issues | No failure possible; add diagnostic warning for structural asymmetry |
| Edge weights 0 for halo in MPI | Known limitation; deferred to MPI parallel plan |
| Non-deterministic across partitions | Known; fix requires hash(global_id); deferred to MPI plan |
